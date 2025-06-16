import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers
from TL_models.deep.multi_layer_dense_network import MultiLayerDense

class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, conv_layers, fc_layers, stride=1, pool_size=2,
                 drop_rate=0.25, BN=True, reg=0.0001, **kwargs):
        super().__init__(**kwargs)
        self.conv_layers_config = conv_layers
        self.fc_layers_config = fc_layers
        self.stride = stride
        self.pool_size = pool_size
        self.drop_rate = drop_rate
        self.BN = BN
        self.reg = reg
        self.feature_extractor = self._build_layers()

    def _build_layers(self):
        layers_list = []

        for filt, kernel in self.conv_layers_config:
            layers_list.append(tf.keras.layers.Conv2D(
                filters=filt,
                kernel_size=kernel,
                strides=self.stride,
                padding='valid',
                activation=None,
                kernel_initializer=tf.keras.initializers.he_normal()
            ))
            if self.BN:
                layers_list.append(tf.keras.layers.BatchNormalization())
            if self.pool_size is not None:
                layers_list.append(tf.keras.layers.MaxPool2D(
                    pool_size=self.pool_size,
                    strides=self.stride
                ))
            layers_list.append(tf.keras.layers.ReLU())

        if len(self.conv_layers_config) > 0:
            layers_list.append(tf.keras.layers.Flatten())

        layers_list.append(MultiLayerDense(self.fc_layers_config,
                                           drop_rate=self.drop_rate,
                                           BN=self.BN,
                                           reg=self.reg))
        return layers_list

    def call(self, inputs, return_all=False):
        x = inputs
        activations = [ ]
        for layer in self.feature_extractor:
            x = layer(x)
            activations.append(x)
        return activations if return_all else x
    