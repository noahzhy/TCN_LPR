from tensorflow.keras import *
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
import tensorflow as tf
import keras


class SeparableConv(Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(SeparableConv, self).__init__(**kwargs)
        self.filters = filters
        out_dim_025 = filters//4
        self.conv1 = Conv2D(out_dim_025, kernel_size=[1, kernel_size],
            padding='same', strides=1, activation='relu6')
        self.conv2 = Conv2D(out_dim_025, kernel_size=[kernel_size, 1],
            padding='same', strides=1, activation='relu6')
        self.conv3 = Conv2D(filters, kernel_size=[1, 1],
            padding='same', strides=1, activation='relu6')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.filters
        return tf.TensorShape(shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
        })
        return config


class DualDilatedBlock(Layer):
    def __init__(self,
                filters,
                dilation_rate=[],
                dropout_rate=0.25,
                activation='relu6',
                **kwargs):
        super(DualDilatedBlock, self).__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate

        self.conv_1x1_in = Conv1D(filters, kernel_size=1, padding="same")
        self.conv_dilated_1 = Conv1D(filters, kernel_size=3, padding="same", dilation_rate=dilation_rate[0], activation='relu6')
        self.conv_dilated_2 = Conv1D(filters, kernel_size=3, padding="same", dilation_rate=dilation_rate[1], activation='relu6')
        self.concat = Concatenate(axis=-1)
        self.dropout = Dropout(dropout_rate)
        self.conv_1x1_out = Conv1D(filters, kernel_size=1, padding='same')
        self.add = Add()

    def call(self, inputs):
        x = shortcut = self.conv_1x1_in(inputs)
        x = self.concat([self.conv_dilated_1(x), self.conv_dilated_2(x)])
        x = self.dropout(x)
        x = self.conv_1x1_out(x)
        x = self.add([x, shortcut])
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "filters": self.filters,
            "dropout_rate": self.dropout_rate,
        })
        return config


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
        self.activation_1 = Activation(activation)
        self.conv_1x1_out = Conv1D(filters, kernel_size=1, padding='same')
        self.dropout_1 = Dropout(rate=dropout_rate)
        self.add = Add()

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
        x = self.conv_1x1_out(x)
        x = self.dropout_1(x, training=training)

        x = self.add([x, skip])
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


class MS_TCN(Layer):
    def __init__(
            self,
            filters: list,
            kernel_size: int = 3,
            depth: int = 6,
            return_sequence: bool = False,
            dropout_rate: float = 0.25,
            activation: str = "relu6",
            **kwargs):

        super(MS_TCN, self).__init__(**kwargs)
        self.blocks = []
        self.depth = depth//2
        self.kernel_size = kernel_size
        self.return_sequence = return_sequence

        for j in range(self.depth):
            dilation_size_conv_1 = 2 ** (self.depth-1-j)
            dilation_size_conv_2 = 2 ** j
            self.blocks.append(
                DualDilatedBlock(
                    filters=filters,
                    dilation_rate=[dilation_size_conv_1, dilation_size_conv_2],
                    name=f"dual_dilated_block_{j}"
                )
            )

        for i in range(self.depth):
            dilation_size = 2 ** i
            self.blocks.append(
                ResidualBlock(
                    filters=filters,
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
        })
        return config
