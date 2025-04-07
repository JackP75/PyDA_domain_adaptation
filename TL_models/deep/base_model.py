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

class BaseModel(tf.keras.models.Model, ABC):
    def __init__(self, params={'feat_fc_layers': [10, 10],
                                'feat_conv_layers': [[10,(3,3)], [10,(5,5)]],#[filters, kernel]
                                'class_layers': [10, 10],
                                'input_dim': 3,
                                'output_size': 3,
                                'drop_rate': 0.25,
                                'reg': 0.0001,
                                'entropy': 1e-6,
                                'BN': True,
                                'lr': 1e-3,
                                'pool_size':2,
                                'stride': 1}, optimiser = keras.optimizers.Adam()):
        super().__init__()

        self.params = params

        # Hyperparameters
        self.output_size = params['output_size']
        self.drop_rate = params['drop_rate']
        self.reg = params['reg']
        self.BN = params['BN']

        self.build_feature_extractor()
        self.build_classifier()
        self.optimiser = optimiser
        self.update = False

    def build_feature_extractor(self):
        # Feature extractor
        self.feature_extractor = []
       
        #  convolutional layers
        for filt, kernel in self.params['feat_conv_layers']:
            self.feature_extractor.append(layers.Conv2D(
                filters=filt,
                kernel_size=kernel,
                strides=self.params['stride'],
                padding='valid',
                activation=None,
                kernel_initializer=tf.keras.initializers.he_normal()
            ))
            
            if self.params['BN']:
                self.feature_extractor.append(layers.BatchNormalization())
            if self.params['pool_size'] is not None:
                self.feature_extractor.append(layers.MaxPool2D(pool_size=self.params['pool_size'], strides=self.params['stride']))
            self.feature_extractor.append(layers.ReLU())

        # flatten #            
        if len(self.params['feat_conv_layers']) > 0:
            self.feature_extractor.append(layers.Flatten())

        # fully connected layers
        for nodes in self.params['feat_fc_layers']:
            self.feature_extractor.append(layers.Dense(
                nodes,
                activation=None,
                kernel_initializer=tf.keras.initializers.he_normal(),
                kernel_regularizer=keras.regularizers.l2(self.reg)
            ))
            
            if self.params['BN']:
                self.feature_extractor.append(layers.BatchNormalization())
            if self.drop_rate > 0:
                self.feature_extractor.append(layers.Dropout(self.drop_rate))
            self.feature_extractor.append(layers.ReLU())

        

    def build_classifier(self):
        self.classifier = []
        for nodes in self.params['class_layers']:

            self.classifier.append(layers.Dense(
                nodes,
                activation=None,
                kernel_initializer=tf.keras.initializers.he_normal(),
                kernel_regularizer=keras.regularizers.l2(self.reg)
            ))
            if self.BN:
                self.classifier.append(layers.BatchNormalization())
            self.classifier.append(layers.Dropout(self.drop_rate))
            self.classifier.append(layers.ReLU())

        # Classifier layer, outputs logits
        self.classifier.append(layers.Dense(
            self.output_size,
            activation=None,
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=keras.regularizers.l2(self.reg)
        ))
   
    def get_feature(self, x_in, training=False, return_all=False):
        x = x_in
        activations = []
        for layer in self.feature_extractor:
            x = layer(x, training=training)
            if 're' in layer.name:
                activations.append(x)

        if return_all and activations[-1] is not x: #need this for flattening layers 
            activations.append(tf.identity(x))

        if return_all:
            return activations 
        else:
            return x
        
    def get_classification_logits(self, z, training=False, return_all=False):
        
        activations = []
        for layer in self.classifier:
            z = layer(z, training=training)
            if 're' in layer.name:
                activations.append(tf.identity(z))

        if return_all and activations[-1] is not z: #need this for flattening layers 
            activations.append(tf.identity(z))

        if return_all:
            return activations
        else:
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
    
    def upsample_data(self, Xs, ys, Xt, yt, source_inds=None, target_inds=None, Ns_OH=0, Nt_OH = 0):
        # Get the sizes of the source and target datasets
        
        size_s = Xs.shape[0] + Ns_OH
        size_t = Xt.shape[0] + Nt_OH 
        # print(size_s, size_t)
        # Determine the smaller and larger dataset sizes
        if size_s == size_t:
            # If both sizes are equal, return the datasets as they are
            return Xt, yt, Xs, ys, source_inds, target_inds
        
        # Identify the smaller dataset
        if size_s < size_t:
            # Upsample the source dataset
            indices = np.random.choice(np.arange(size_s - Ns_OH), size=size_t - Ns_OH, replace=True)
            Xs_upsampled = Xs[indices]
            ys_upsampled = ys[indices]
            if source_inds is not None:
                source_inds = source_inds[indices]
            
            return Xt, yt, Xs_upsampled, ys_upsampled, source_inds, target_inds
        else:
            # Upsample the target dataset
            indices = np.random.choice(np.arange(size_t - Nt_OH), size=size_s - Nt_OH, replace=True)
            Xt_upsampled = Xt[indices]
            if yt is not None:
                yt_upsampled = yt[indices]
            else:
                yt_upsampled = None

            if target_inds is not None:
                target_inds = target_inds[indices]
            
            return Xt_upsampled, yt_upsampled, Xs, ys, source_inds, target_inds
    
    
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
        
        return f1,acc

    def plot_loss(self):
        # Plot loss
        epochs = len(self.class_loss)
        n = range(1, math.floor(epochs) + 1, 1)
        plt.plot(n, self.domain_loss, label=self.params['loss_type'])
        plt.plot(n, self.class_loss, label="class loss")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

        # Plot F1
        plt.plot(n, self.f1, label="F1 per 10 epochs")
        plt.xlabel('epoch')
        plt.ylabel('x_test F1')
        plt.legend()
        plt.show()

    def apply_dimensionality_reduction(self, Zs, Zt, feature_reduction):
        """Apply the specified dimensionality reduction technique."""
        # Standardize the features
        scale = StandardScaler()
        Z = np.vstack((Zs, Zt))
        Z = scale.fit(Z).transform(Z)

        if feature_reduction == 'PCA':
            pca = PCA()
            pca.fit(Z)
            Zs = pca.transform(Zs)
            Zt = pca.transform(Zt)
            print(f'PCA explained variance ratio cumulative sum: {np.cumsum(pca.explained_variance_ratio_)[:10]}')
            return Zs, Zt

        elif feature_reduction == 't-SNE':
            tsne = TSNE(n_components=2, random_state=42)
            Zs = tsne.fit_transform(Zs)
            Zt = tsne.fit_transform(Zt)
            return Zs, Zt

        else:
            raise ValueError("Unsupported feature reduction method. 'PCA' and 't-SNE' are implemented.")

    def plot_transfer(self, Zs, Zt, ys, yt, colour='domain', name=None, xlabel='PC 1', ylabel='PC 2'):
        """Plot the transfer visualization with label numbers as markers, 
        domain represented by colour, and optional feature dimensions."""

        plt.figure(figsize=(10, 8))

        for i, point in enumerate(Zs):
            plt.text(point[0], point[1], str(ys[i]), fontsize=12, ha='center', va='center', color='red')  

        for i, point in enumerate(Zt):
            plt.text(point[0], point[1], str(yt[i]), fontsize=12, ha='center', va='center', color='blue') 

        if colour == 'domain':
            plt.scatter(Zs[:, 0], Zs[:, 1], c=ys, label='Source Domain', alpha=0.0, cmap='viridis')
            plt.scatter(Zt[:, 0], Zt[:, 1], c=yt, label='Target Domain', marker='x', alpha=0.0, cmap='viridis')
        else:
            plt.scatter(Zs[:, 0], Zs[:, 1], c=ys, label='Source Domain', alpha=0.0)
            plt.scatter(Zt[:, 0], Zt[:, 1], c=yt, label='Target Domain', alpha=0.0)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

        if name:
            plt.title(name)
        plt.show()


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

    def measure_divergence(self, test_data, args):
        """Measure divergence between source and target domains 
        using specified method (PAD, MMD, or JMMD).
        Specifying None for lengeth scale used the median heuristic"""
    
        Xt = test_data[0]
        Xs = test_data[2]
        yt = test_data[1]
        ys = test_data[3]
        Yt = np.unique(yt).shape[0]
        Ys = np.unique(ys).shape[0]
        
        # Get the features (assumed to be the last element from self.get_feature)
        Zs = self.get_feature(Xs)[-1].numpy()
        Zt = self.get_feature(Xt)[-1].numpy()

        method = args.get('method', 'PAD')  
        kernel = args.get('kernel', 'rbf')  
        kernel = args.get('l', None)  

        l = args.get('l', None) 
        if method == 'PAD':
            return PAD(Zs, Zt, kernel=kernel, Ys=Ys, Yt=Yt)
        elif method == 'MMD':
            return MMD(Zs, Zt, l=l)
        elif method == 'JMMD':
            return JMMD(Zs, Zt, ys, yt, l=l)   
        else:
            raise ValueError(f"Only 'PAD', 'MMD', and 'JMMD' are implemented.")


    def get_summary(self):
        self.built = True
        self.summary()

