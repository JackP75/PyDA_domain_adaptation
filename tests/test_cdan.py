import pytest
import tensorflow as tf
import os, sys

cwd = os.getcwd()
parts = cwd.split(os.sep)
pyda_index = parts.index("PyDA") + 1
pyda_path = os.sep.join(parts[:pyda_index])
sys.path.append(pyda_path)
from TL_models.deep.CDAN import CDAN_model
from TL_models.deep.utils import soft_loss

@pytest.fixture
def model():
    params={'feat_conv_layers': [],#[filters, kernel]
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
        'stride': (1,1)}
    return CDAN_model(params)

@pytest.fixture
def sample_data():
    batch_size = 10
    feature_nodes = 10  # Must match params['feature_nodes']
    output_size = 4  # Must match params['output_size']

    Xs = tf.random.normal([batch_size, feature_nodes])
    Xt = tf.random.normal([batch_size, feature_nodes])
    class_indices = tf.random.uniform([batch_size], minval=0, maxval=output_size, dtype=tf.int32)
    ys = tf.one_hot(class_indices, depth=output_size)

    return Xs, Xt, ys

def test_get_multi_map_shape(model, sample_data):
    Xs, _, _ = sample_data

    feat_activations = tf.random.normal([Xs.shape[0], model.params['feat_fc_layers'][-1]])
    class_activations = tf.random.normal([Xs.shape[0], model.params['output_size']])

    multi_map = model.get_multi_map(feat_activations, class_activations)

    expected_shape = (Xs.shape[0], model.params['feat_fc_layers'][-1] * model.params['output_size'])
    assert multi_map.shape == expected_shape, f"Expected shape {expected_shape}, got {multi_map.shape}"

def test_gradient_behavior(model, sample_data):
    Xs, Xt, ys = sample_data

    with tf.GradientTape() as tape:
        ds = model(Xs, train=True)
        dt = model(Xt, train=True)
        disc_loss = model.get_disc_loss(ds, dt)
    
    classifier_trainable = [var for layer in model.classifier for var in layer.trainable_variables]
    gradients_disc = tape.gradient(disc_loss, classifier_trainable)

    assert all(g is None for g in gradients_disc), "Classifier gradients should be stopped in get_disc_loss."

    with tf.GradientTape() as tape:
        ds = model(Xs, train=True)
        dt = model(Xt, train=True)

        class_loss = soft_loss(ds[-1], ys)  # Compute classification loss
        disc_loss = model.get_disc_loss(ds, dt)

    gradients_class = tape.gradient(class_loss, classifier_trainable)

    assert any(g is not None for g in gradients_class), "Classifier gradients should be active in class_loss."

if __name__ == "__main__":

    print("Running CDAN tests...")
    pytest.main(["-v", __file__])