"""
Implementation of the joint adaptation network (JAN) presented in:
Deep Transfer Learning with Joint Adaptation Networks, Mingsheng Long et al. (2017)

@author: Jack Poole
"""

import tensorflow as tf
from TL_models.dist_measures import median_heuristic
from TL_models.deep.DAN import DAN_model


#the issue was the way lmda was being fed into the gradrev layer =)
class JAN_model(DAN_model):
    def __init__(self,params={'feature_nodes':10,
                              'num_feat_layers':2,
                              'num_class_layers':2,
                              'output_size':3 ,
                              'drop_rate':0.25,
                              'reg':0.0001,
                              'entropy':0,
                              'BN':True,
                              'lr':1e-3,
                              'kernel':'rbf',
                              'length scale':None,
                              'adapt_extractor':False,
                              'type':'dense'}, optimiser = tf.keras.optimizers.Adam()):
        super().__init__(params, optimiser)   
        
        self.params=params
        
        #hyperparameters
        self.entropy=params['entropy']
    
    def linear_kernel_samples(self, zs, zt, l=None, type_flag='square'):
        
        ns = tf.shape(zs)[0]

        if l is None:
            l = median_heuristic(zs, zt, type_flag=type_flag)  # Compute bandwidth

        # Generate index pairs for efficient computation

        even_inds = tf.range(0, ns, delta=2)
        odd_inds = tf.range(1, ns, delta=2)
        zs1, zs2 = zs[odd_inds], zs[even_inds]
        zt1, zt2 = zt[odd_inds], zt[even_inds]

        Kss = self.pair_rbf(zs1, zs2, l)
        Ktt = self.pair_rbf(zt1, zt2, l)
        Kst = self.pair_rbf(zs1, zt2, l)
        Kts = self.pair_rbf(zt1, zs2, l)

        return Kss, Ktt, Kst, Kts
    
    def linear_MMD_loss(self, Zs, Zt):
        #this is the JMMD defined in eq(11)
        ns = tf.shape(Zs[0])[0]

        KSS, KTT, KTS, KST = tf.ones([ns//2,]), tf.ones([ns//2,]), tf.ones([ns//2,]), tf.ones([ns//2,])
        for layerN, z in enumerate(zip(Zs, Zt)):
            zs, zt = z
            for lsN, l in enumerate(self.length_scales):
                Kss, Ktt, Kst, Kts = self.linear_kernel_samples(zs, zt, l=l)
                KSS *= Kss
                KTT *= Ktt
                KST *= Kst
                KTS *= Kts
        loss = 2.0 / ns * tf.reduce_sum(KSS + KTT - KST - KTS)
        return loss
    
    def update_MK_MMD(self,X,Y=None):
        #does not actually update MK-MMD, just uses this name to fit with DAN

        tf.experimental.numpy.experimental_enable_numpy_behavior()
        if self.params['length scales'] is None or self.params['length scales'] == 'median':
            Zs = self.get_feature(X, return_all=True)
            Zt = self.get_feature(Y, return_all=True)     
            source_feat_activations, source_class_activations =self.call(X,train=True)
            target_feat_activations, target_class_activations =self.call(Y,train=True)
            Zs =  source_feat_activations[self.N_conv:] + source_class_activations #don't match conv laters
            Zt = target_feat_activations[self.N_conv:] + target_class_activations
            self.length_scales= [median_heuristic(zs, zt) for zs, zt in zip(Zs, Zt)]
        else:
            assert len(self.params['length scales']) == (len(self.params['class_layers']) + len(self.params['feat_fc_layers'])), \
            f"Mismatch in length scales: expected {len(self.params['class_layers']) + len(self.params['feat_fc_layers'])}, but got {len(self.params['length scales'])}"
            self.length_scales = self.params['length scales']
       
            

