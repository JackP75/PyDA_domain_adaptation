"""
Implementation of the domain adaptation network (DAN) presented in:
Learning transferable features with deep adaptation networks, M Long et al. (2015)

@author: Jack Poole
"""

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from TL_models.dist_measures import median_heuristic, MMD
from TL_models.standard_checks import check_inputs
from TL_models.deep.utils import soft_loss,get_lambda
from TL_models.deep.base_model import BaseModel

from cvxopt import matrix, solvers
from sklearn.metrics import  pairwise_distances
import tensorflow.experimental.numpy as tnp

class DAN_model(BaseModel):
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
                            'stride': 1}, optimiser=tf.keras.optimizers.Adam()):
        super().__init__(params, optimiser)   
         

        self.params=params
        
        #hyperparameters
        self.lr=params['lr']
        self.entropy=params['entropy']
        self.N_conv = len(self.params['feat_conv_layers'])
        if self.N_conv >0:
            self.N_conv += 1 #flatten layer

    def call(self, x_in, train=False):
       
        #feature extractor
        feat_activations=self.get_feature(x_in,training=train, return_all=train)   
        #classifier
        if train:
            feat_in = feat_activations[-1]
        else:
            feat_in = feat_activations
        class_activations=self.get_classification_logits(feat_in,training=train, return_all=train)
        
        return (feat_activations, class_activations)
        
    def rbf(self, X1, X2, l):
        #WARNING: this version is only for linear_mmd and assumes X1, X2 are single samples
        
        X1, X2 = X1.reshape(1,-1), X2.reshape(1,-1)
        K=np.exp(-np.square(pairwise_distances(X1, X2, metric='euclidean')) / (2*l**2))
        return K
    
    def gk(self, Xs, Xt, i, l):
       
        g = self.rbf(Xs[2*i-1], Xs[2*i], l) + self.rbf(Xt[2*i-1], Xt[2*i], l) \
            - self.rbf(Xt[2*i-1], Xs[2*i], l) - self.rbf(Xs[2*i-1], Xt[2*i], l)
        
        return g
        
    def delta_gk(self, Xs, Xt, i, l):
        return self.gk(Xs, Xt, 2*i - 1, l ) - self.gk(Xs, Xt, 2*i, l )
    
    def pair_rbf(self, X, Y, l):
        return tf.exp(-tf.norm(X - Y, axis=1) ** 2 / (2.0 * l ** 2))

    def linear_MMD(self, zs, zt, l=None, type_flag='square'):

        ns = tf.shape(zs)[0]

        if l is None:
            l = median_heuristic(zs, zt, type_flag=type_flag)  # Compute bandwidth

        # Generate index pairs for efficient computation

        even_inds = tf.range(0, ns, delta=2)
        odd_inds = tf.range(1, ns, delta=2)
        min_len = min(len(even_inds), len(odd_inds))
        even_inds = even_inds[:min_len]
        odd_inds = odd_inds[:min_len]
        zs1, zs2 = zs[odd_inds], zs[even_inds]
        zt1, zt2 = zt[odd_inds], zt[even_inds]

        Kss = self.pair_rbf(zs1, zs2, l)
        Ktt = self.pair_rbf(zt1, zt2, l)
        Kst = self.pair_rbf(zs1, zt2, l)
        Kts = self.pair_rbf(zt1, zs2, l)

        mmd = 2 / ns * tf.reduce_sum(Kss + Ktt - Kst - Kts)

        return mmd
    
    def get_Q(self, zs, zt):
        
        ns = tf.shape(zs)[0]
        nt = tf.shape(zs)[0]

        even_inds = np.arange(0, ns, 2)
        odd_inds = np.arange(1, ns, 2)
        min_len = min(len(even_inds), len(odd_inds))
        even_inds = even_inds[:min_len]
        odd_inds = odd_inds[:min_len]
        zs1, zs2 = zs[even_inds], zs[odd_inds]

        even_inds = np.arange(0, nt, 2)
        odd_inds = np.arange(1, nt, 2)
        min_len = min(len(even_inds), len(odd_inds))
        even_inds = even_inds[:min_len]
        odd_inds = odd_inds[:min_len]
        zt1, zt2 = zt[even_inds], zt[odd_inds]

        even_inds = np.arange(0, ns//2, 2)
        odd_inds = np.arange(1, ns//2, 2)
        min_len = min(len(even_inds), len(odd_inds))
        even_inds = even_inds[:min_len]
        odd_inds = odd_inds[:min_len]

        m = len(self.length_scales)
        Q = np.zeros((m,m))
        for m1,l1 in enumerate(self.length_scales):
            for m2,l2 in enumerate(self.length_scales):
                Kss = self.pair_rbf(zs1, zs2, l1)
                Ktt = self.pair_rbf(zt1, zt2, l1)
                Kst = self.pair_rbf(zs1, zt2, l1)
                Kts = self.pair_rbf(zt1, zs2, l1)
                gk1 = Kss + Ktt - Kst - Kts
                gk1_delta = gk1[odd_inds] - gk1[even_inds]

                Kss = self.pair_rbf(zs1, zs2, l2)
                Ktt = self.pair_rbf(zt1, zt2, l2)
                Kst = self.pair_rbf(zs1, zt2, l2)
                Kts = self.pair_rbf(zt1, zs2, l2)
                gk2 = Kss + Ktt - Kst - Kts
                gk2_delta = gk2[odd_inds] - gk2[even_inds]
                Q[m1,m2] = tf.reduce_sum(gk1_delta * gk2_delta)
        return Q * 4/ns
    
    def linear_MMD_loss(self, Zs, Zt):
        loss = 0
        for layerN, z in enumerate(zip(Zs, Zt)):
            zs, zt = z
            for lsN, l in enumerate(self.length_scales):
                if float(self.betas[layerN][lsN]) > 1.0e-4: ### skip calulations if length scale has small contribution
                    loss += self.betas[layerN][lsN] * self.linear_MMD(zs, zt, l=l)
                else:
                    loss +=  tf.constant(0.0, dtype=tf.float64)
        return loss
    
    def update_MK_MMD(self,X,Y=None):
        
        tf.experimental.numpy.experimental_enable_numpy_behavior()
        if self.params['length scales'] is None:
            self.length_scales= [2**p for p in np.arange(-8, 8.5, 0.5)]#used in paper
        else:
            self.length_scales = self.params['length scales']
        
        m = len(self.length_scales)
        ##get features
        Zs = self.get_feature(X, return_all=True)
        Zt = self.get_feature(Y, return_all=True)     
        source_feat_activations, source_class_activations =self.call(X,train=True)
        target_feat_activations, target_class_activations =self.call(Y,train=True)
        Zs =  source_feat_activations[self.N_conv:] + source_class_activations #don't match conv laters
        Zt = target_feat_activations[self.N_conv:] + target_class_activations

        if m == 1:
            self.betas = [[1] for i in Zs]
        elif self.params['length scales'] == 'median':
            
            self.length_scales = [median_heuristic(zs, zt) for zs, zt in zip(Zs, Zt)]

            self.betas = []
            for i in range(len(Zs)):
                temp = np.zeros([m,])
                temp[i] = 1
                self.betas.append(temp)
        else:
            print('Estimating MK-MMD coefficients...')
            #its somewhat ambiguous how they deal with multiple layers
            #here assuming each layer has indepedent kernel weights
            self.betas = []
            for zs, zt in zip(Zs, Zt):
                zs, zt = tnp.array(zs), tnp.array(zt)

                d =np.zeros((m,1))
                for ind, l in enumerate(self.length_scales):
                    d[ind] = self.linear_MMD(zs, zt, l=l)
                
                Q = self.get_Q(zs, zt) + 1e-6 * np.eye(m)
                # print(Q)
                # print(d)     
                Q = matrix(Q)
                d = matrix(d)
                q=matrix(np.zeros((m,1)))
                G=matrix(-np.eye(m))#
                h=matrix(np.zeros((m,1)))#
                A=matrix(d.T) #this would be add to one
                b=matrix(1.0) 
                
                solvers.options['show_progress'] = False
                try:
                    beta = solvers.qp(Q, q, G, h, A, b)   
                    B = beta['x']
                    # print(f'beta: {B}')     
                    self.betas.append(tf.nn.softmax(np.array(beta['x']).astype('float32'),axis=0))
                except:
                    print('Failed')
                    self.betas.append(1/m*tf.ones([m,1],dtype='float32'))

    def get_valid_loss(self,test_data):
        Xt=test_data[0]
        Xs=test_data[2][:Xt.shape[0],:]
        C=np.unique(test_data[3]).shape[0]
        ys=tf.one_hot(test_data[3],C)[:Xt.shape[0],:]
        source_activation=self.call(Xs)
        target_activation=self.call(Xt)
        class_loss=soft_loss(source_activation[-1],ys)
        MMD_loss=self.get_MMD_loss(source_activation,target_activation)
        return class_loss,MMD_loss
        
    @tf.function 
    def train_step(self,Xs,ys,Xt,lmda=1.0):
        with tf.GradientTape() as tape:

            source_feat_activations, source_class_activations =self.call(Xs,train=True)
            target_feat_activations, target_class_activations =self.call(Xt,train=True)
            class_loss=soft_loss(source_class_activations[-1],ys)

            adapt_source_layers =  source_feat_activations[self.N_conv:] + source_class_activations #don't match conv laters
            adapt_target_layers = target_feat_activations[self.N_conv:] + target_class_activations
            MMD_loss = self.linear_MMD_loss(adapt_source_layers, adapt_target_layers)
            pt=tf.nn.softmax(target_class_activations[-1])
            h=tf.multiply(pt,tf.math.log(pt))            
            target_entropy=-tf.reduce_mean(h,axis=1)
            loss = class_loss + lmda * MMD_loss + tf.add_n(self.losses) + self.entropy*target_entropy
    
        #get gradients 
        gradients= tape.gradient(loss, self.trainable_variables)
       
        #update
        #print(self.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.trainable_variables)) #zips together lists creating list of tuples (a_i,b_i)
        
        return class_loss,MMD_loss
        
    def fit(self, Xs, ys, Xt, yt=None, s_OH=None, t_OH=None, source_inds=None, target_inds=None,
             epochs=10, pretrain=False, batch_size = 100, print_training=False, lmda=None):
        
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
                if lmda is None:
                    lmda =  tf.constant(get_lambda(progress), dtype=tf.float32) #
                else:
                    pass

            if epoch % 10 == 0: 
                self.update_MK_MMD(Xs, Xt)
                # tf.numpy_function(self.update_MK_MMD(Xs[:100],Xt[:100]))
                # print(self.betas)
            for batch in dataset:
                if yt is None:
                    Xs,ys,Xt = batch
                    class_loss,disc_loss =self.train_step(Xs,ys,Xt, 
                                                        lmda=lmda)
                else:
                    Xs,ys,Xt, yt = batch
                    class_loss,disc_loss, class_losst =self.train_step(Xs,ys,Xt, yt, 
                                                                       lmda=lmda, source_inds=source_inds, target_inds=target_inds)
                c_class_loss+=class_loss
                # c_classt_loss+=class_losst
                c_disc_loss+=disc_loss
            c_lossL.append(c_class_loss)
            disc_lossL.append(c_disc_loss)
            ct_lossL.append(c_classt_loss)
            
            if epoch%10==0: 
                #print metrics
               
                if self.update:
                    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                    if ys is None:
                        print('Source classifier loss: '+str(c_class_loss)+'| MMD loss:'+ str(c_disc_loss))
                    else:
                        print('Source classifier loss: '+str(c_class_loss)+'| MMD loss:'+ str(c_disc_loss)+'| Target classifier loss:'+ str(c_classt_loss))
            
        if print_training:
            plt.plot(range(epochs), c_lossL)
            plt.plot(range(epochs), disc_lossL)
            plt.legend(['cross-entropy loss', 'MMD loss'])
            if yt is not None:            
                plt.plot(range(epochs), ct_lossL)
                plt.legend(['cross-entropy loss', 'MMD loss', 'target cross-entropy loss'])
            plt.show()
