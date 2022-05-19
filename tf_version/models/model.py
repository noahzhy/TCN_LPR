import sys

sys.path.append('./')
import tensorflow.keras as keras
import keras.backend as K
import tensorflow as tf
from config import *
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers.optimizer_v2 import *
from keras.optimizers import *
from keras_flops import get_flops

from models.ms_tcn import MS_TCN
from models.stn import STN
from models.tcn import TCN


class CTCLoss(Layer):
    def __init__(self, name=None, **kwargs):
        super(CTCLoss, self).__init__(name=name, **kwargs)
        self.loss_fn = K.ctc_batch_cost

    def call(self, y_true, y_pred):
        input_length = K.tile([[K.shape(y_pred)[1]]], [K.shape(y_pred)[0], 1])
        label_length = K.tile([[K.shape(y_true)[1]]], [K.shape(y_true)[0], 1])
        return self.loss_fn(y_true, y_pred, input_length, label_length)


class CEM(Layer):
    def __init__(self, filters, **kwargs):
        super(CEM, self).__init__(**kwargs)
        self.filters = filters
        self.conv1x1_0 = Conv2D(filters, kernel_size=[1, 1], strides=[1, 1], padding='same')
        self.conv1x1_1 = Conv2D(filters, kernel_size=[1, 1], strides=[1, 1], padding='same')
        # self.conv1x1_2 = Conv2D(filters, kernel_size=[1, 1], strides=[1, 1], padding='same')
        self.upsample_x2 = UpSampling2D(size=(2,2))
        self.upsample_x4 = UpSampling2D(size=(4,4))
        self.add = Add()

    def call(self, inputs):
        t1, t2, t3 = inputs
        t1 = self.conv1x1_0(t1)
    
        t2 = self.conv1x1_1(t2)
        t2 = self.upsample_x2(t2)

        # t3 = self.conv1x1_2(t3)
        t3 = self.upsample_x4(t3)

        return self.add([t1, t2, t3])

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


class PCM(Layer):
    def __init__(self, filters, **kwargs):
        super(PCM, self).__init__(**kwargs)
        self.filters = filters
        self.conv1x1_0 = Conv2D(filters, kernel_size=[1, 1], strides=[1, 1], padding='same')
        self.conv1x1_1 = Conv2D(filters, kernel_size=[1, 1], strides=[1, 1], padding='same')
        # self.conv1x1_2 = Conv2D(filters, kernel_size=[1, 1], strides=[1, 1], padding='same')
        self.ap_x1 = AvgPool2D(pool_size=(2, 2), padding='same')
        self.ap_x2 = AvgPool2D(pool_size=(4, 4), padding='same')
        self.add = Add()

    def call(self, inputs):
        t1, t2, t3 = inputs
        t1 = self.ap_x2(t1)
        t1 = self.conv1x1_0(t1)
    
        t2 = self.ap_x1(t2)
        t2 = self.conv1x1_1(t2)

        return self.add([t1, t2, t3])

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


class GCM(Layer):
    def __init__(self, pool_size, **kwargs):
        super(GCM, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.ap = AveragePooling2D(pool_size=pool_size, padding='same')

    def call(self, inputs):
        x = pool = self.ap(inputs)
        # x = tf.reduce_mean(tf.square(x))
        # x = tf.compat.v1.div(pool, x)
        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        # shape[-1] = self.filters
        return tf.TensorShape(shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
        })
        return config


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


class Separable_Conv(Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(Separable_Conv, self).__init__(**kwargs)
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

    x = Conv2D(64, kernel_size=[3, 3], strides=[2, 1], padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu6')(x)
    x = MaxPool2D(strides=[1, 1], padding='SAME')(x)

    x = Separable_Conv(128, kernel_size=3, name='separable_conv_1')(x)
    t1, _ = tf.split(x, num_or_size_splits=2, axis=1)
    x = MaxPool2D(padding='SAME')(x)

    x = Separable_Conv(128, kernel_size=3, name='separable_conv_2')(x)
    t2, _ = tf.split(x, num_or_size_splits=2, axis=1)
    x = MaxPool2D(padding='SAME')(x)

    x = Separable_Conv(256, kernel_size=3, name='separable_conv_3')(x)
    t3, bottom = tf.split(x, num_or_size_splits=2, axis=1)
    # x = MaxPool2D(padding='SAME')(x)

    # x = Separable_Conv(256, kernel_size=3, name='separable_conv_4')(x)
    # x = Conv2D(256, kernel_size=[2, 1], padding='same')(x)

    # x = MaxPool2D(padding='SAME')(x)

    # g1 = GCM(4)(t1)
    # g2 = GCM(2)(t2)
    # g3 = GCM(1)(t3)
    # top = Concatenate(axis=-1)([g1, g2, g3])
    
    top = PCM(256)([t1, t2, t3]) 

    x = Concatenate(axis=2)([top, bottom])

    x = MS_TCN(128, kernel_size=3, depth=8)(x)

    x = Dense(NUM_CLASS, kernel_initializer='he_normal',
              activation='softmax', name='softmax0')(x)

    output = CTCLoss()(labels, x)
    model = Model(inputs=[inputs, labels], outputs=output, name='TCN_LPR')
    return model


if __name__ == '__main__':
    model = TCN_LPR()
    model.summary()
    # flops = get_flops(model, batch_size=1)
    # print(f"FLOPS: {flops / 10 ** 9:.03} G")
