import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers


class MultiLayerDense(tf.keras.layers.Layer):
    def __init__(
        self,
        layer_sizes,
        activation='ReLU',
        BN=False,
        drop_rate=0.0,
        reg=0.0,
        final_units=None,
        final_activation=None,
        name=None
    ):
        """
        Multi-layer dense block for feature extraction or regression/classification. 
        The activations of each layer (MUST BE RELU) are returned if return_all is True.

        Args:
            layer_sizes: List[int], number of units in each hidden Dense layer.
            activation: Activation function for hidden layers (e.g., 'relu').
            BN: Bool, whether to apply BatchNormalization after Dense layers.
            drop_rate: Float, dropout rate (0.0 means no dropout).
            reg: Float, L2 regularization coefficient.
            final_units: Optional[int], units in the final output layer (e.g., number of classes).
            final_activation: Optional[str], activation for the final layer (e.g., 'softmax').
            name: Optional[str], layer name.
        """
        super(MultiLayerDense, self).__init__(name=name)
        self.return_activations = []
        self.nn = []

        for i, units in enumerate(layer_sizes):
            self.nn.append(layers.Dense(
                units,
                activation=None,
                kernel_initializer=initializers.he_normal(),
                kernel_regularizer=regularizers.l2(reg),
                name=f'dense_{i}'
            ))

            if BN:
                self.nn.append(layers.BatchNormalization(name=f'bn_{i}'))

            if drop_rate > 0:
                self.nn.append(layers.Dropout(drop_rate, name=f'dropout_{i}'))

            self.nn.append(getattr(layers, activation)(name=f'{activation}_{i}'))

        if final_units is not None:
            self.nn.append(layers.Dense(
                final_units,
                activation=final_activation,
                kernel_initializer=initializers.he_normal(),
                kernel_regularizer=regularizers.l2(reg),
                name='final_output'
            ))

    def call(self, inputs, training=False, return_all=False):
        x = inputs
        outputs = []

        for layer in self.nn:
            x = layer(x, training=training) if hasattr(layer, 'training') else layer(x)
            if return_all and isinstance(layer, layers.Layer) and self.activation in layer.name:
                outputs.append(x)

        return outputs if return_all else x

