
"""
Implementation of the domain adversarial neural network (DANN) presented in:
Domain-Adversarial Training of Neural Networks, Ganin et al. (2015)

@author: Jack Poole
"""


import time
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import GradReverse, soft_loss, soft_loss2, sig_loss, get_lambda
from .base_model import BaseModel
from TL_models.standard_checks import check_inputs
from TL_models.deep.multi_layer_dense_network import MultiLayerDense

class DANN_model(BaseModel):
    def __init__(self, training_data, training_params={'lr': 1e-3, 
                                'optimiser':tf.keras.optimizers.SGD(),
                                'epochs': 100,
                                'batch_size': 32,
                                'update':False, 
                                'pretrain': False},
                 model_params={'feat_fc_layers': [10, 10],
                            'feat_conv_layers': [[10,(3,3)], [10,(5,5)]],#[filters, kernel]
                            'class_layers': [10, 10],
                            'disc_layers':[10, 10],
                            'input_dim': 3,
                            'output_size': 3,
                            'drop_rate': 0.25,
                            'reg': 0.0001,
                            'entropy': 1e-6,
                            'BN': True,
                            'lr': 1e-3,
                            'pool_size':2,
                            'stride': 1}):
        super().__init__(training_data, training_params, model_params)

        self.reverse = GradReverse()
        self.discriminator = MultiLayerDense(self.model_params['disc_layers'],
                                              drop_rate=self.model_params['drop_rate'],
                                              BN=self.model_params['BN'],
                                              reg=self.model_params['reg'],
                                              final_units=2,
                                              final_activation=None)


    def call(self, x_in, lmda=tf.constant(1.0), train=False):

        #feature extractor
        feat_activations=self.get_feature(x_in, training=train)

        #classifier
        class_activations=self.get_classification_logits(feat_activations, training=train)
    
        #domain discriminator
        d=self.reverse(feat_activations, lmda)
        Dlogit = self.discriminator(d, training=train)
        
        return (Dlogit, class_activations) #returns logits of discriminator and classifier
    
    def get_disc_loss(self,ds,dt):
   
        yd=tf.concat([ds[0],dt[0]],0)
        domain = tf.concat([tf.tile([[1., 0.]], [tf.shape(ds[0])[0], 1]),
                            tf.tile([[0., 1.]], [tf.shape(dt[0])[0], 1])], axis=0)
        disc_loss = soft_loss(yd,domain)
        return disc_loss
    
    def get_disc_f1(self,test_data):
        Xt=test_data[0]
        Xs=test_data[2]
        ds=self.call(Xs)
        dt=self.call(Xt)
        yd=tf.concat([ds[0],dt[0]],0)
        domain= np.vstack([np.tile([1], [ds[0].shape[0], 1]),
                               np.tile([0], [dt[0].shape[0], 1])]).astype('float32')
        y_pred=np.argmax(yd,axis=1)
        f1=f1_score(domain, y_pred,average='micro')
        return f1
    
    def get_valid_loss(self,test_data):
        Xt=test_data[0]
        Xs=test_data[2]
        C=self.params['output_size']
        ys=tf.one_hot(test_data[3],C)
        source_activation=self.call(Xs)
        target_activation=self.call(Xt)
        class_loss=soft_loss(source_activation[-1],ys)
        disc_loss=self.get_disc_loss(source_activation,target_activation)
        return class_loss,disc_loss
   
    @tf.function 
    def train_step(self,Xs,ys,Xt, yt=None,lmda=tf.constant(1.0)):
        class_losst = 0
        with tf.GradientTape() as tape:

            ds =self.call(Xs,lmda=lmda,train=True)
            dt =self.call(Xt,lmda=lmda,train=True)

            #cross-entropy loss
            class_loss=soft_loss(ds[-1],ys)
            
            #entropy in target
            pt=tf.nn.softmax(dt[-1],axis=1)
            h=tf.multiply(pt,tf.math.log(pt))            
            target_entropy=-tf.reduce_mean(h,axis=0)

            #discriminator loss
            disc_loss =self.get_disc_loss(ds,dt)
                         
            #target cross entropy loss for semi-supervised
            if yt is not None and self.T_sig:
                class_losst = soft_loss2(dt[-1], yt, self.target_inds)
                class_loss =  self.Q * class_losst + (1-self.Q) * class_loss #weight the mean of losses
                loss = class_loss + lmda * disc_loss + tf.add_n(self.losses) + self.model_params['entropy'] * target_entropy
            else:
                loss = class_loss + lmda * disc_loss + tf.add_n(self.losses) + self.model_params['entropy'] * target_entropy

        #get gradients 
        gradients= tape.gradient(loss, self.trainable_variables)
       
        #update
        self.optimiser.apply_gradients(zip(gradients, self.trainable_variables)) 
        
        return class_loss,disc_loss, class_losst
    
        
    def fit(self):
        
        epochs = self.training_params['epochs']
        pretrain = self.training_params.get('pretrain', False)
        update = self.training_params.get('update', False)
        self.optimiser.learning_rate.assign(self.training_params['lr'])

        c_lossL, disc_lossL, ct_lossL = [], [], [] #tracking total loss per epoch
        for epoch in tqdm(range(epochs)):
            #init cumulative loss 
            c_class_loss=0
            c_classt_loss=0
            c_disc_loss =0
            
            start= time.time()

            #set hyperparameters for epoch
            if pretrain:
                lmda = tf.constant(0.0, dtype=tf.float32)
                self.T_sig = False
            else:
                self.T_sig = True
                lmda =  tf.constant(get_lambda(epoch/epochs), dtype=tf.float32) 
  
            for batch in self.training_dataset:
                if not self.semi_super:
                    Xs,ys,Xt = batch
                    class_loss,disc_loss, class_losst =self.train_step(Xs,ys,Xt, 
                                                                       lmda=lmda)
                else:
                    Xs,ys,Xt, yt = batch
                    class_loss, disc_loss, class_losst =self.train_step(Xs,ys,Xt, yt, 
                                                                       lmda=lmda)
                c_class_loss+=class_loss
                c_classt_loss+=class_losst
                c_disc_loss+=disc_loss
            c_lossL.append(c_class_loss)
            disc_lossL.append(c_disc_loss)
            ct_lossL.append(c_classt_loss)
            
            if epoch%100==0: 
                #print metrics
               
                if update:
                    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                    if yt is None:
                        print('Source classifier loss: '+str(c_class_loss)+'| discriminator loss:'+ str(c_disc_loss))
                    else:
                        print('Source classifier loss: '+str(c_class_loss)+'| discriminator loss:'+ str(c_disc_loss)+'| Target classifier loss:'+ str(c_classt_loss))
            
        return c_lossL, disc_lossL, ct_lossL
