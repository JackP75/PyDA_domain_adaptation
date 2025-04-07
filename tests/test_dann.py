import pytest
import tensorflow as tf
import numpy as np
import sys, os

cwd = os.getcwd()
parts = cwd.split(os.sep)
pyda_index = parts.index("PyDA") + 1
pyda_path = os.sep.join(parts[:pyda_index])
sys.path.append(pyda_path)
from TL_models.deep.DANN import DANN_model 
from TL_models.deep.utils import GradReverse, soft_loss, soft_loss2, sig_loss

def test_grad_reverse():
    x = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    lmda =  tf.constant(0.5)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = GradReverse()(x, lmda)  # Forward pass
    grad = tape.gradient(y, x)  # Compute gradient
    assert np.allclose(y.numpy(), x.numpy()), "Forward pass should be identity"
    assert np.allclose(grad.numpy(), -lmda * np.ones_like(x.numpy())), "Gradient should be reversed and scaled"

def test_model_structure():
    params={'feat_conv_layers': [[32,(5,5)], [48,(5,5)]],#[filters, kernel]
        'feat_fc_layers': [],
        'class_layers': [100, 100],
        'disc_layers':[100],
        'input_dim': 3,
        'output_size': 10,
        'drop_rate': 0.25,
        'reg': 1e-3,
        'entropy': 0,
        'BN': True,
        'lr': 1e-2,
        'pool_size':(2,2),
        'stride': (1,1), 
        'length scales':None}
    
    
    model = DANN_model(params)

    assert len(model.feature_extractor) > 0, "Feature extractor should not be empty"
    assert len(model.classifier) == (len(params['class_layers'])*4+1), f"Classifier layers mismatch: {len(model.classifier)} vs {(len(params['class_layers'])+1)}"

    assert len(model.discriminator) == (len(params['disc_layers'])*4 ), f"Discriminator layers should be {len(params['disc_layers'])*4 + 1} but got {len(model.discriminator)}"
    assert model.disc_out.units == 2, "Discriminator output should have 2 units"
    assert model.classifier[-1].units == 10, f"Classifier output should have {params['output_size']} units"

def test_loss_functions():
    y_pred = tf.constant([[0.8, 0.2], [0.4, 0.6]], dtype=tf.float32)
    y_true = tf.constant([[1, 0], [0, 1]], dtype=tf.float32)
    
    assert soft_loss(y_pred, y_true) >= 0, "Soft loss should be non-negative"
    assert soft_loss2(y_pred, y_true, tf.constant([1, 1])) >= 0, "Soft loss with mask should be non-negative"
    assert sig_loss(y_pred, y_true) >= 0, "Sigmoid loss should be non-negative"
    assert soft_loss2(y_pred, y_true, tf.constant([1, 1])) == soft_loss(y_pred, y_true), "Masked should be the same"

def test_domain_classifier():
    model = DANN_model(params={'feat_conv_layers': [],#[filters, kernel]
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
 
    Dlogit, Clogit = model(X)
    Dlogit, Clogit = model(X)
    assert Dlogit.shape == (5, 2), f"Domain classifier should output (batch size, 2) logits but got { Dlogit.shape}"
    assert Clogit.shape == (5, 4), f"Classifer should have (bactch size, number of classes)  but got { Clogit.shape}"

def test_train_step_updates_weights():
    model = DANN_model(params={'feat_conv_layers': [],#[filters, kernel]
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
    Xs = tf.random.normal((10, 100))
    ys = tf.one_hot(np.random.randint(0, 4, 10), 4)
    Xt = tf.random.normal((10, 100))
    
    model(Xs)
    initial_weights = [w.numpy().copy() for w in model.trainable_variables]
    model.train_step(Xs, ys, Xt)
    updated_weights = [w.numpy().copy() for w in model.trainable_variables]
    
    weight_changes = [not np.allclose(w1, w2) for w1, w2 in zip(initial_weights, updated_weights)]
    assert any(weight_changes), "At least one weight should change after a training step"

if __name__ == "__main__":

    print("Running DANN tests...")
    pytest.main(["-v", __file__])