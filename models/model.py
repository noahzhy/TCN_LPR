# from tensorflow.keras import *
from keras.callbacks import *
from keras.optimizer_v2.adam import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
import tensorflow as tf
import keras
import sys
sys.path.append('./')

from models.tcn import TCN
from models.stn import STN
from config import *

chars = CHARS
char_map = {chars[c]: c for c in range(len(chars))} # 验证码编码（0到len(chars) - 1)
idx_map = {value: key for key, value in char_map.items()} # 编码映射到字符
idx_map[-1] = '' # -1映射到空

def ctc_loss(args):
    return K.ctc_batch_cost(*args)


def ctc_decode(softmax):
    return K.ctc_decode(softmax, K.tile([K.shape(softmax)[1]], [K.shape(softmax)[0]]))[0][0]


def char_decode(label_encode): 
    return [''.join([idx_map[column] for column in row]) for row in label_encode]


class CTCLoss(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = K.ctc_batch_cost

    def call(self, y_true, y_pred):
        # batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        # label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        # input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")

        # label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        # input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        # loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        # self.add_loss(loss)
        # return y_pred

        labels = y_true
        x = y_pred
        label_length = Lambda(lambda x: K.tile([[K.shape(x)[1]]], [K.shape(x)[0], 1]))(labels)
        input_length = Lambda(lambda x: K.tile([[K.shape(x)[1]]], [K.shape(x)[0], 1]))(x)
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        return loss


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
        self.conv = Conv2D(filters, kernel_size, padding='same', strides=strides)
        self.batchNorm = BatchNormalization(axis=-1)
        self.relu = ReLU(max_value=6.)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batchNorm(x)
        x = self.relu(x)
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
    def __init__(self, filters):
        super(FlattenedConv, self).__init__()
        self.filters = filters
        outdim_025 = filters//4
        self.conv0 = Conv2D(outdim_025, kernel_size=(1,3), padding='same', strides=1, activation='relu')
        self.conv1 = Conv2D(outdim_025, kernel_size=(3,1), padding='same', strides=1, activation='relu')
        self.conv2 = Conv2D(filters, kernel_size=(1,1), padding='same', strides=1, activation='relu')

    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
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
    n, h, w, c = x.get_shape().as_list()
    top, bottom = tf.split(x, 2, axis=1)
    x = concatenate([top, bottom], axis=2)
    return x


def TCN_LPR():
    x = inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL), batch_size=BATCH_SIZE, name='input_image', dtype="float32")
    # x = STN()(x)
    labels = Input(shape=(None,), batch_size=BATCH_SIZE, name="label", dtype="int64")

    x = Conv2D(64, kernel_size=(3,3), padding='same', strides=2, activation='relu')(x)
    x = MaxPool2D(strides=(1,1), name='maxpool0', padding='SAME')(x)

    x = FlattenedConv(128)(x)
    x = MaxPool2D(padding='SAME')(x)

    x = FlattenedConv(128)(x)
    x = MaxPool2D(padding='SAME')(x)

    x = FlattenedConv(256)(x)
    x = multi_line(x)

    # x = Dropout(rate=0.5)(x)

    x = TCN([64, 64, 64, 64], kernel_size=3)(x)
    # x = logits = tf.reduce_mean(x, axis=2)
    x = Dense(NUM_CLASS, kernel_initializer='he_normal', activation='softmax', name='softmax0')(x)
    output = CTCLoss()(labels, x)
    model = Model(inputs=[inputs, labels], outputs=output, name='TCN_LPR')
    model.compile(
        loss=lambda y_true, y_pred: y_pred,
        optimizer=Adam(learning_rate=LEARNING_RATE),
    )

    return model


if __name__ == '__main__':
    model = TCN_LPR()
    model.summary()
