# from tensorflow.keras import *
import sys
sys.path.append('./')
from config import *
from models.stn import STN
from models.tcn import TCN
from models.ms_tcn import MS_TCN
from keras.callbacks import *
from keras.optimizer_v2.adam import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
import tensorflow as tf
import keras
from keras_flops import get_flops


class CTCLoss(Layer):
    def __init__(self, name=None, **kwargs):
        super(CTCLoss, self).__init__(name=name, **kwargs)
        self.loss_fn = K.ctc_batch_cost

    def call(self, y_true, y_pred):
        input_length = K.tile([[K.shape(y_pred)[1]]], [K.shape(y_pred)[0], 1])
        label_length = K.tile([[K.shape(y_true)[1]]], [K.shape(y_true)[0], 1])
        return self.loss_fn(y_true, y_pred, input_length, label_length)


class ConvBlock(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="SAME",
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.kernel_size = kernel_size
        self.conv = Conv2D(filters, kernel_size,
                           padding='same', strides=strides)
        self.batchNorm = BatchNormalization(axis=-1)
        self.relu6 = ReLU(max_value=6.)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batchNorm(x)
        x = self.relu6(x)
        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = shape[1] // self.strides
        shape[2] = shape[2] // self.strides
        shape[-1] = self.filters
        return tf.TensorShape(shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
        })
        return config


class FlattenedConv(Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(FlattenedConv, self).__init__(**kwargs)
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


def multi_line(x):
    top, bottom = tf.split(x, 2, axis=1)
    x = concatenate([top, bottom], axis=2)
    return x, top, bottom


def TCN_LPR():
    # x = STN(name='stn')(inputs)
    x = inputs = Input(
        shape=(HEIGHT, WIDTH, CHANNEL),
        name='image',
        dtype="float32",
    )
    labels = Input(shape=(None,), name="label", dtype="int64")

    x = Conv2D(64, kernel_size=[3, 3], strides=[2, 1],
        padding='same', activation='relu6')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(strides=[1, 1], padding='SAME')(x)

    x = FlattenedConv(128, kernel_size=3, name='flattened_conv1')(x)
    x = MaxPool2D(padding='SAME')(x)

    x = FlattenedConv(128, kernel_size=3, name='flattened_conv2')(x)
    x = MaxPool2D(padding='SAME')(x)

    x = FlattenedConv(256, kernel_size=3, name='flattened_conv3')(x)
    top, bottom = tf.split(x, num_or_size_splits=2, axis=1)

    ## last version
    # top = TCN([32]*4, kernel_size=3)(top)
    # x = TCN([64]*6, kernel_size=3)(x)
    top = MS_TCN(32, kernel_size=3, depth=4)(top)
    x = MS_TCN(64, kernel_size=3, depth=6)(x)
    x = Concatenate(axis=2)([top, x])

    x = Dense(NUM_CLASS, kernel_initializer='he_normal',
              activation='softmax', name='softmax0')(x)

    output = CTCLoss()(labels, x)
    model = Model(inputs=[inputs, labels], outputs=output, name='TCN_LPR')
    return model


if __name__ == '__main__':
    model = TCN_LPR()
    model.summary()
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
