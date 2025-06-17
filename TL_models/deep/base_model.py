"""
Base class for domain adaptation architectures.
"""

import time, math
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score

from TL_models.dist_measures import PAD, MMD, JMMD
from TL_models.deep.multi_layer_dense_network import MultiLayerDense
from TL_models.deep.feature_extractor import FeatureExtractor

from TL_models.standard_checks import check_inputs


class BaseModel(tf.keras.models.Model, ABC):
    def __init__(self, training_data,
                       training_params = {'lr': 1e-3, 
                                          'optimiser':keras.optimizers.Adam(),
                                          'epochs': 100,
                                          'batch_size': 32,
                                          'update':False, 
                                          'pretrain': False}, 
                       model_params={'feat_fc_layers': [10, 10],
                                'feat_conv_layers': [[10,(3,3)], [10,(5,5)]],
                                'class_layers': [10, 10],
                                'input_dim': 3,
                                'output_size': 3,
                                'drop_rate': 0.25,
                                'reg': 0.0001,
                                'entropy': 1e-6,
                                'BN': True,
                                'pool_size': 2,
                                'stride': 1} 
                                ):
        super().__init__()

        """"Base class for domain adaptation architectures.
        Args: 
        training data: tuple of (Xs, ys, Xt, yt) where yt is optional. 
                       For semi-supervised DA, yt should be None to indicate missing labels.
        training_params: dictionary of training parameters including learning rate, optimiser, and update flag.
        model_params: dictionary of model parameters to define architecture
        """

        self.model_params = model_params
        self.training_params = training_params
        self.optimiser = training_params['optimiser']

        self.build_model()
        self.prepare_data(training_data) #ensures data is same size and is in tensorflow format

    def build_model(self):
        self.feature_extractor = FeatureExtractor(self.model_params['feat_conv_layers'],
                            self.model_params['feat_fc_layers'],
                            drop_rate=self.model_params['drop_rate'],
                            BN=self.model_params['BN'],
                            reg=self.model_params['reg']
                           )
        self.classifier = MultiLayerDense(self.model_params['class_layers'],
                            drop_rate=self.model_params['drop_rate'],
                            BN=self.model_params['BN'],
                            reg=self.model_params['reg'],
                            final_units=self.model_params['output_size'],
                            final_activation=None)

    def get_feature(self, x_in, training=False, return_all=False):
        x = self.feature_extractor(x_in, return_all=return_all, training=training)
        return x
       
    def get_classification_logits(self, z, training=False, return_all=False):
        z = self.classifier(z, return_all=return_all, training=training)
        return z
    
    @abstractmethod
    def call(self, x_in, train=False):
        return NotImplementedError
    
    @abstractmethod
    def train_step(self):
        return NotImplementedError
    
    @abstractmethod
    def fit(self):
        return NotImplementedError

    @staticmethod
    def upsample_data(Xs, ys, Xt, yt, Nlabelled_target=0):
        # Get the sizes of the source and target datasets
        
        size_s = Xs.shape[0]
        size_t = Xt.shape[0] + Nlabelled_target 

        # Determine the smaller and larger dataset sizes
        if size_s == size_t:
            # If both sizes are equal, return the datasets as they are
            return Xt, yt, Xs, ys
        
        # Identify the smaller dataset
        if size_s < size_t:
            # Upsample the source dataset
            indices = np.random.choice(np.arange(size_s), size=size_t, replace=True)
            Xs_upsampled = Xs[indices]
            ys_upsampled = ys[indices]
            
            return Xt, yt, Xs_upsampled, ys_upsampled
        else:
            # Upsample the target dataset
            indices = np.random.choice(np.arange(size_t), size=size_s, replace=True)
            Xt_upsampled = Xt[indices]
            if yt is not None:
                yt_upsampled = yt[indices]
            else:
                yt_upsampled = None
        
            return Xt_upsampled, yt_upsampled, Xs, ys
        
    def prepare_data(self, training_data):
        """Prepare data for training/testing, 
        assuming data was orginally numpy arrays with categorical label encoding"""

        if len(training_data) == 3:
            Xs, ys, Xt = training_data
            yt = None
            self.semi_super = False
        elif len(training_data) == 4:
            Xs, ys, Xt, yt = training_data
            self.semi_super = True
        else:
            raise ValueError('Training data must be a tuple of (Xs, ys, Xt) or (Xs, ys, Xt, yt)!')

        #check data is correct shape
        check_inputs(Xs, ys, Xt, yt)

        if yt is not None:
            #prepare data for semi-supervised DA
            label_index = np.where(yt is not None)[0]
            #ratio of target labelled to source data is used for weighting losses
            self.Q = label_index.shape[0] / (label_index.shape[0] + ys.shape[0])
        else:
            Nlabelled_target = 0  
            

        Xt ,yt, Xs, ys = self.upsample_data(Xs, ys, Xt, yt, Nlabelled_target)
       
        u = np.unique(ys.reshape(-1)).shape[0]
        self.Ns = ys.shape[0]
        ys=tf.one_hot(ys, u)
        if yt is not None: #semi-supervised DA
            self.target_inds = np.where(yt is not None)[0]
            yt=tf.one_hot(yt, u)
            self.training_dataset = tf.data.Dataset.from_tensor_slices((Xs, ys, Xt, yt)).shuffle(200).batch(self.training_params['batch_size'])
        else: #unsupervised DA
            self.training_dataset = tf.data.Dataset.from_tensor_slices((Xs, ys, Xt)).shuffle(200).batch(self.training_params['batch_size'])


    def predict(self,x):
        logit_C=self.call(x)[-1] #networks have multiple outputs; lasts is always class logits
        return np.argmax(tf.nn.softmax(logit_C,axis=1), axis=1)
    
    def predict_proba(self,x):
        logit_C=self.call(x)[-1]
        return np.array(tf.nn.softmax(logit_C, axis=1))

    def entropy(self, X_test, classes = None):

        if classes is None:
            classes= [c for c in range(self.num_classes)]
        probs = self.predict_proba(X_test)
        epsilon = 1e-9
        probs = np.clip(probs, epsilon, 1 - epsilon)
        return  -np.sum(probs[:, classes] * np.log(probs[:, classes]), axis=1)
    
    def get_performance(self,x,y):
        #labels assumed to be in categorical 0,1,2 etc
        y_pred=self.predict(x)
        acc=accuracy_score(y,y_pred)
        f1=f1_score(y, y_pred,average='macro')
        
        return f1, acc

    def get_summary(self):
        self.built = True
        self.summary()

    def from_json(cls, config_file):
        """Loads data and parameters from a JSON file."""
        pass

    def plot_feature(self, test_data, colour='domain', name=None, feature_reduction='PCA'):
            """Main method to plot feature reduction."""
            
            Xt = test_data[0]
            yt = test_data[1]
            Xs = test_data[2]
            ys = test_data[3]
            
            Zs = self.get_feature(Xs).numpy()
            Zt = self.get_feature(Xt).numpy()
            
            Zs, Zt = self.apply_dimensionality_reduction(Zs, Zt, feature_reduction)

            self.plot_transfer(Zs, Zt, ys, yt, colour=colour, name=name, xlabel='$X_1$', ylabel='$X_2$')
