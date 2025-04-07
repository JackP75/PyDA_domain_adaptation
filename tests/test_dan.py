import pytest
import tensorflow as tf
import numpy as np
import sys, os

cwd = os.getcwd()
parts = cwd.split(os.sep)
pyda_index = parts.index("PyDA") + 1
pyda_path = os.sep.join(parts[:pyda_index])
sys.path.append(pyda_path)
from TL_models.deep.DAN import DAN_model 

def test_pair_rbf_kernel():
    model = DAN_model()
    X1 = tf.random.normal([5, 5])
    X2 = tf.random.normal([5, 5])
    l = 1.0

    K = model.pair_rbf(X1, X2, l)

    assert tf.reduce_all(K >= 0)  # Kernel values should be positive
    assert K.shape == (5,)  # Expecting a scalar per pair

def test_MK_MMD():
    model = DAN_model(params={'feat_conv_layers': [],#[filters, kernel]
        'feat_fc_layers': [10],
        'class_layers': [10],
        'disc_layers':[],
        'input_dim': 3,
        'output_size': 4,
        'drop_rate': 0.25,
        'reg': 1e-3,
        'entropy': 0,
        'BN': True,
        'lr': 1e-2,
        'pool_size':(2,2),
        'stride': (1,1), 
        'length scales':None})
    
    Xs = tf.random.normal((100, 10)) + 5
    Xt = tf.random.normal((100, 10)) - 5

    model.update_MK_MMD(Xs, Xt)
    Zs, Zt = model(Xs), model(Xt)   
    mmd_loss = model.linear_MMD_loss(Zs, Zt)

    assert np.allclose([np.sum(betas) for betas in model.betas], np.ones((len(model.betas)))), f"Betas do not sum to 1"
    assert mmd_loss >= 0, f"MMD loss should be non-negative but got {mmd_loss}"

def test_classifier():
    model = DAN_model(params={'feat_conv_layers': [],#[filters, kernel]
        'feat_fc_layers': [100],
        'class_layers': [100, 100],
        'disc_layers':[100],
        'input_dim': 3,
        'output_size': 4,
        'drop_rate': 0.25,
        'reg': 1e-3,
        'entropy': 0,
        'BN': True,
        'lr': 1e-2,
        'pool_size':(2,2),
        'stride': (1,1), 
        'length scales':None})
    X = tf.random.normal((5, 10))  
 
    _, Clogit = model(X)
    _, Clogit = model(X)
    assert Clogit.shape == (5, 4), f"Classifer should have (bactch size, number of classes)  but got { Clogit.shape}"

def test_train_step_updates_weights():
    model = DAN_model(params={'feat_conv_layers': [],#[filters, kernel]
        'feat_fc_layers': [10],
        'class_layers': [10],
        'disc_layers':[],
        'input_dim': 3,
        'output_size': 4,
        'drop_rate': 0.25,
        'reg': 1e-3,
        'entropy': 0,
        'BN': True,
        'lr': 1e-2,
        'pool_size':(2,2),
        'stride': (1,1), 
        'length scales':[1,10]})
    
    Xs = tf.random.normal((10, 5))
    ys = tf.one_hot(np.random.randint(0, 4, 10), 4)
    Xt = tf.random.normal((10, 5))
    model.update_MK_MMD(Xs, Xt)
    initial_weights = [w.numpy().copy() for w in model.trainable_variables]
    model.train_step(Xs, ys, Xt)
    updated_weights = [w.numpy().copy() for w in model.trainable_variables]
    
    weight_changes = [not np.allclose(w1, w2) for w1, w2 in zip(initial_weights, updated_weights)]
    assert any(weight_changes), "At least one weight should change after a training step"

if __name__ == "__main__":

    print("Running DAN tests...")
    pytest.main(["-v", __file__])