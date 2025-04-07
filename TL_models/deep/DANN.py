
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

class DANN_model(BaseModel):
    def __init__(self,params={'feat_fc_layers': [10, 10],
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
                            'stride': 1}, optimiser = keras.optimizers.Adam()):
        super().__init__(params, optimiser)    

        #hyperparameters
        self.reg=params['reg']
        self.BN=params['BN']
        self.entropy=params['entropy']
        self.lr=params['lr']
        self.update = False

        self.build_discriminator()

        
    def build_discriminator(self):
        #domain discriminator
        self.reverse=GradReverse()
        self.discriminator=[]
        for nodes in self.params['disc_layers']:
            self.discriminator.append(layers.Dense(nodes, activation=None, 
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   kernel_regularizer=keras.regularizers.l2(self.reg)))
            if self.BN:
                self.discriminator.append(layers.BatchNormalization())
            self.discriminator.append(layers.Dropout(self.drop_rate))
            self.discriminator.append(layers.ReLU())
        self.disc_out=layers.Dense(2,activation=None)

    def call(self, x_in, lmda=tf.constant(1.0), train=False):
       
        #feature extractor
        feat_activations=self.get_feature(x_in,training=train)   

        #classifier
        class_activations=self.get_classification_logits(feat_activations,training=train)
    
        #domain discriminator
        d=self.reverse(feat_activations, lmda)
        for layer in self.discriminator:
            d=layer(d)
        Dlogit=self.disc_out(d)
        
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
    def train_step(self,Xs,ys,Xt, yt=None,lmda=tf.constant(1.0), source_inds=None, target_inds=None):
        class_losst = 0
        with tf.GradientTape() as tape:

            ds =self.call(Xs,lmda=lmda,train=True)
            dt =self.call(Xt,lmda=lmda,train=True)

            if source_inds is None:
                class_loss=soft_loss(ds[-1],ys)
            else:
                #if some source are unlabelled
                class_loss = soft_loss2(ds[-1], ys, source_inds)
                       
            pt=tf.nn.softmax(dt[-1],axis=1)
            h=tf.multiply(pt,tf.math.log(pt))            
            target_entropy=-tf.reduce_mean(h,axis=0)
            disc_loss =self.get_disc_loss(ds,dt)
                         
            #add optional target loss for semi-supervised
            if yt is not None and self.T_sig:
                class_losst = soft_loss2(dt[-1], yt, target_inds)
                class_loss =  self.Q * class_losst + (1-self.Q) * class_loss #weight the mean of losses
                loss = class_loss + lmda * disc_loss + tf.add_n(self.losses) + self.entropy*target_entropy
            else:
                loss = class_loss + disc_loss + tf.add_n(self.losses) + self.entropy*target_entropy #lmda * 

        #get gradients 
        gradients= tape.gradient(loss, self.trainable_variables)
       
        #update
        self.optimiser.apply_gradients(zip(gradients, self.trainable_variables)) 
        
        return class_loss,disc_loss, class_losst
    
        
    def fit(self, Xs, ys, Xt, yt=None, s_OH=None, t_OH=None, source_inds=None, target_inds=None,
             epochs=10, pretrain=False, batch_size = 100, print_training=False):
        
        check_inputs(Xs, ys, Xt, yt)
        
        if s_OH is not None:
            Xs_OH, ys_OH = s_OH
            Ns_OH = ys_OH.shape[0]

        else:
            Ns_OH = 0

        if t_OH is not None:
            Xt_OH, yt_OH = t_OH
            Nt_OH = yt_OH.shape[0]
        else:
            Nt_OH = 0
        
        u = np.unique(ys.reshape(-1)).shape[0]
        self.Ns = ys.shape[0]

        if yt is not None and (source_inds is None or target_inds is None):
            raise ValueError('To train in semi-supervised setting, specify source_inds/target_inds mask vectors!')
        elif yt is not None:
             #proportion of source/target labelled data for weighting losses
            self.Q = np.sum(target_inds) / (np.sum(target_inds) + np.sum(source_inds))

        Xt ,yt, Xs, ys, source_inds, target_inds = self.upsample_data(Xs, ys, Xt, yt, source_inds,
                                                                       target_inds, Ns_OH, Nt_OH)
        ys=tf.one_hot(ys, u)
        
        if s_OH is not None:
            Xs = tf.concat([Xs, Xs_OH], axis=0)
            ys = tf.concat([ys, ys_OH], axis=0)
            source_inds = tf.concat([source_inds, tf.zeros([ys_OH.shape[0]])], axis=0)
        
        if yt is not None: #semi-supervised DA
            yt=tf.one_hot(yt, u)
            if t_OH is not None:
                Xt = tf.concat([Xt, Xt_OH], axis=0)
                yt = tf.concat([yt, yt_OH], axis=0)
                target_inds = tf.concat([target_inds, tf.zeros([yt_OH.shape[0]])], axis=0)
            dataset = tf.data.Dataset.from_tensor_slices((Xs,ys, Xt, yt)).shuffle(200).batch(batch_size)

        else: #unsupervised DA
            dataset = tf.data.Dataset.from_tensor_slices((Xs,ys, Xt)).shuffle(200).batch(batch_size)
    
        self.optimiser.learning_rate.assign(self.lr)
        c_lossL, disc_lossL, ct_lossL = [], [], []
        for epoch in tqdm(range(epochs)):
            #init cumulative loss 
            c_class_loss=0
            c_classt_loss=0
            c_disc_loss =0
            
            start= time.time()

            #set hyperparameters for epoch
            progress=epoch/epochs
            if pretrain:
                lmda = tf.constant(0.0, dtype=tf.float32)
                self.T_sig = False
            else:
                self.T_sig = True
                lmda =  tf.constant(get_lambda(progress), dtype=tf.float32) #
  
            for batch in dataset:
                if yt is None:
                    Xs,ys,Xt = batch
                    class_loss,disc_loss, class_losst =self.train_step(Xs,ys,Xt, 
                                                                       lmda=lmda, source_inds=source_inds, target_inds=target_inds)
                else:
                    Xs,ys,Xt, yt = batch
                    class_loss,disc_loss, class_losst =self.train_step(Xs,ys,Xt, yt, 
                                                                       lmda=lmda, source_inds=source_inds, target_inds=target_inds)
                c_class_loss+=class_loss
                c_classt_loss+=class_losst
                c_disc_loss+=disc_loss
            c_lossL.append(c_class_loss)
            disc_lossL.append(c_disc_loss)
            ct_lossL.append(c_classt_loss)
            
            if epoch%100==0: 
                #print metrics
               
                if self.update:
                    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                    if yt is None:
                        print('Source classifier loss: '+str(c_class_loss)+'| discriminator loss:'+ str(c_disc_loss))
                    else:
                        print('Source classifier loss: '+str(c_class_loss)+'| discriminator loss:'+ str(c_disc_loss)+'| Target classifier loss:'+ str(c_classt_loss))
            
        if print_training:
            plt.plot(range(epochs), c_lossL)
            plt.plot(range(epochs), disc_lossL)
            plt.legend(['cross-entropy loss', 'discrimator loss'])
            if yt is not None:            
                plt.plot(range(epochs), ct_lossL)
                plt.legend(['cross-entropy loss', 'discrimator loss', 'target cross-entropy loss'])
            plt.show()
