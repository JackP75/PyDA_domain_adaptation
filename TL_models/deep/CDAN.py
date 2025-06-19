
"""
Implementation of the conditional domain adversarial neural network (CDAN) presented in:
Conditional Adversarial Domain Adaptation, Mingsheng Long et al. (2018)

@author: Jack Poole
"""

import tensorflow as tf
from .DANN import DANN_model

#the issue was the way lmda was being fed into the gradrev layer =)
class CDAN_model(DANN_model):
    def __init__(self, training_data, 
                 training_params={'lr': 1e-3, 
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
   