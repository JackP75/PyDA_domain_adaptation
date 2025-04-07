
"""
Implementation of the conditional domain adversarial neural network (CDAN) presented in:
Conditional Adversarial Domain Adaptation, Mingsheng Long et al. (2018)

@author: Jack Poole
"""

import tensorflow as tf
from .DANN import DANN_model

#the issue was the way lmda was being fed into the gradrev layer =)
class CDAN_model(DANN_model):
    def __init__(self,params={'feature_nodes':10,
                              'num_feat_layers':2,
                              'num_class_layers':2,
                              'disc_nodes':20,
                              'num_disc_layers':2,
                              'output_size':3,
                              'drop_rate':0.25,
                              'reg':0.0001,
                              'entropy':1e-6,
                              'BN':True,
                              'lr':1e-3}, optimiser=tf.keras.optimizers.Adam()):
    
        super().__init__(params, optimiser)    
        
    def get_multi_map(self,feat_activations,class_activations):
        yc_exp=tf.stop_gradient(tf.expand_dims(tf.nn.softmax(class_activations),1))
        feature_exp=tf.expand_dims(feat_activations,2)
        outer_prod=tf.raw_ops.BatchMatMul(x=feature_exp,y=yc_exp)
        return tf.reshape(outer_prod, [feat_activations.shape[0], -1])
    
    def call(self, x_in, lmda=tf.constant(1.0), train=False):
       
        #feature extractor
        feat_activations=self.get_feature(x_in,training=train)   

        #classifier
        class_activations=self.get_classification_logits(feat_activations,training=train)
            
        multi_map=self.get_multi_map(feat_activations,class_activations)
        d=self.reverse(multi_map,lmda)
        for layer in self.discriminator:
            d=layer(d)
        Dlogit=self.disc_out(d)
        
        return (Dlogit,class_activations) #returns logits
   