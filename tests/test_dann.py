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

# Dummy dataset
X = tf.random.normal((10, 28, 28, 3))  # e.g., MNISTM format with 3 channels
y = tf.one_hot(np.random.randint(0, 10, 10), 10)
dataset = (X, y, X)  # (source_X, source_y, target_X)

training_params = {
    'lr': 1e-3,
    'optimiser': tf.keras.optimizers.SGD(),
    'epochs': 1,
    'batch_size': 4,
    'update': False,
    'pretrain': True
}

model_params = {
    'feat_fc_layers': [10],
    'feat_conv_layers': [[8, (3, 3)]],
    'class_layers': [10],
    'disc_layers': [10],
    'input_dim': 3,
    'output_size': 10,
    'drop_rate': 0.25,
    'reg': 1e-3,
    'entropy': 0,
    'BN': True,
    'pool_size': 2,
    'stride': 1
}

def test_grad_reverse():
    x = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    lmda = tf.constant(0.5)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = GradReverse()(x, lmda)
    grad = tape.gradient(y, x)
    assert np.allclose(y.numpy(), x.numpy()), "Forward pass should be identity"
    assert np.allclose(grad.numpy(), -lmda.numpy() * np.ones_like(x.numpy())), "Gradient should be reversed and scaled"

def test_model_structure():
    model = DANN_model(dataset, training_params, model_params)

    assert len(model.feature_extractor) > 0, "Feature extractor should not be empty"
    assert model.classifier[-1].units == model_params['output_size'], \
        f"Classifier output units mismatch: expected {model_params['output_size']} but got {model.classifier[-1].units}"
    assert model.disc_out.units == 2, "Domain classifier output should have 2 units"

def test_loss_functions():
    y_pred = tf.constant([[0.8, 0.2], [0.4, 0.6]], dtype=tf.float32)
    y_true = tf.constant([[1, 0], [0, 1]], dtype=tf.float32)

    assert soft_loss(y_pred, y_true) >= 0, "Soft loss should be non-negative"
    assert soft_loss2(y_pred, y_true, tf.constant([1, 1])) >= 0, "Soft loss2 should be non-negative"
    assert sig_loss(y_pred, y_true) >= 0, "Sigmoid loss should be non-negative"
    assert soft_loss2(y_pred, y_true, tf.constant([1, 1])) == soft_loss(y_pred, y_true), "Loss values should match"

def test_domain_classifier():
    model = DANN_model(dataset, training_params, model_params)
    X_test = tf.random.normal((5, 28, 28, 3))

    Dlogit, Clogit = model(X_test)
    assert Dlogit.shape == (5, 2), f"Domain classifier output shape mismatch: got {Dlogit.shape}"
    assert Clogit.shape == (5, model_params['output_size']), f"Classifier output shape mismatch: got {Clogit.shape}"

def test_train_step_updates_weights():
    model = DANN_model(dataset, training_params, model_params)
    Xs = tf.random.normal((10, 28, 28, 3))
    ys = tf.one_hot(np.random.randint(0, 10, 10), 10)
    Xt = tf.random.normal((10, 28, 28, 3))

    model(Xs)
    initial_weights = [w.numpy().copy() for w in model.trainable_variables]
    model.train_step(Xs, ys, Xt)
    updated_weights = [w.numpy().copy() for w in model.trainable_variables]

    weight_changes = [not np.allclose(w1, w2) for w1, w2 in zip(initial_weights, updated_weights)]
    assert any(weight_changes), "At least one weight should change after a training step"

if __name__ == "__main__":
    print("Running DANN model tests...")
    pytest.main(["-v", __file__])
