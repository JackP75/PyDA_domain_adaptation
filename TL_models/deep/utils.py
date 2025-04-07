"""
Helper functions for training nueral networks 
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import  pairwise_distances


#gradient reverse layer
@tf.custom_gradient
def grad_reverse(x,lmda=1.0):
    y = tf.identity(x)
    def grad(dy):
        return lmda*-dy, None
    return y, grad

class GradReverse(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x,lmda=1.0):
        return grad_reverse(x,lmda)
    
    def get_config(self):
        config={}#not sure if this will work

        
def soft_loss(yPred,yTrue):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yPred, labels=yTrue))

def soft_loss2(yPred, yTrue, inds):
    # Calculate the softmax cross-entropy loss with some labels masked
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=yPred, labels=yTrue)
    mask = tf.cast(inds == 1, tf.float32)  # Convert to float for masking
    masked_loss = loss * mask
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

def sig_loss(yPred,yTrue):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yPred, labels=yTrue))

#lmda scheduler
def get_lambda(progress):
    return 2. / (1+np.exp(-10.*progress)) - 1. 


def median_heuristic(X1, X2, type_flag='square'):
    
    if type_flag == 'square':
        'v = E([||x_i-x_j||] | 1 ≤ i < j ≤ n) from \"A Kernel Two-Sample Test, Gretton et al.\"'
        return 0.5*np.median(np.square(pairwise_distances(X1, X2)))**(1/2)
    elif type_flag == 'abs':
        'v =Sqrt(H_n); H_n = E([x_i-x_j]^2 | 1 ≤ i < j ≤ n) from \"Large sample analysis of the median heuristic, Garreau et al.\"'
        return np.median(pairwise_distances(X1, X2))