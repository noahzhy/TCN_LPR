from tensorflow.keras import *
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
import tensorflow as tf
import keras


class ResidualBlock(Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 dilation_rate: int,
                 dropout_rate: float,
                 activation: str,
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.filters = filters
        self.causal_conv_1 = Conv1D(
            filters=self.filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            use_bias=False)
        self.weight_norm_1 = LayerNormalization()
        self.dropout_1 = Dropout(rate=dropout_rate)
        self.activation_1 = Activation(activation)

        self.causal_conv_2 = Conv1D(
            filters=self.filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            use_bias=False)
        self.weight_norm_2 = LayerNormalization()
        self.dropout_2 = Dropout(rate=dropout_rate)
        self.activation_2 = Activation(activation)

        self.activation_3 = Activation(activation)

    def build(self, input_shape):
        in_channels = input_shape[-1]
        if in_channels == self.filters:
            self.skip_conv = None
        else:
            self.skip_conv = Conv1D(
                filters=self.filters,
                kernel_size=1,
                padding='same',
            )
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        if self.skip_conv is None:
            skip = inputs
        else:
            skip = self.skip_conv(inputs)

        x = self.causal_conv_1(inputs)
        x = self.weight_norm_1(x)
        x = self.activation_1(x)
        x = self.dropout_1(x, training=training)

        x = self.causal_conv_2(x)
        x = self.weight_norm_2(x)
        x = self.activation_2(x)
        x = self.dropout_2(x, training=training)

        x = self.activation_3(x + skip)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            # 'kernel_size': self.kernel_size,
            # 'dilation_rate': self.dilation_rate,
            # 'dropout_rate': self.dropout_rate,
            # 'activation': self.activation,
        })
        return config


class TCN(Layer):
    def __init__(
            self,
            filters: list,
            kernel_size: int = 3,
            return_sequence: bool = False,
            dropout_rate: float = 0.25,
            activation: str = "relu",
            **kwargs):

        super(TCN, self).__init__(**kwargs)
        self.blocks = []
        self.depth = len(filters)
        self.kernel_size = kernel_size
        self.return_sequence = return_sequence

        for i in range(self.depth):
            dilation_size = 2 ** i
            self.blocks.append(
                ResidualBlock(
                    filters=filters[i],
                    kernel_size=kernel_size,
                    dilation_rate=dilation_size,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    name=f"residual_block_{i}"
                )
            )

        if not self.return_sequence:
            self.slice_layer = Lambda(lambda tt: tt[:, -1, :])

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        for block in self.blocks:
            x = block(x)

        if not self.return_sequence:
            x = self.slice_layer(x)
        return x

    @property
    def receptive_field_size(self):
        return receptive_field_size(self.kernel_size, self.depth)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'blocks': self.blocks,
            'depth': self.depth,
            'kernel_size': self.kernel_size,
            'return_sequence': self.return_sequence,
            # 'dropout_rate': self.dropout_rate,
            # 'activation': self.activation,
        })
        return config
