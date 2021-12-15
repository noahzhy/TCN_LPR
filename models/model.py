from tensorflow.keras import *
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
import tensorflow as tf
import keras

from models.tcn import TCN
from models.stn import STN
from config import *


class CTCLoss(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


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


def multi_line(x):
    n, h, w, c = x.get_shape().as_list()
    top, bottom = tf.split(x, 2, axis=1)
    # miniside = w//4
    # top = top[:, :, miniside:miniside*3, :]
    x = concatenate([top, bottom], axis=2)
    # x = Resizing(height=h, width=w)(x)
    return x


def TCN_LPR(inputs):
    x = inputs
    x = multi_line(x)
    labels = Input(shape=(None,), batch_size=BATCH_SIZE, name="label")

    x = Conv2D(32, kernel_size=(3,3), padding='same', strides=2, activation='relu6')(x)
    x = MaxPool2D()(x)
    x = Conv2D(64, kernel_size=(5,5), padding='same', strides=2, activation='relu6')(x)
    x = MaxPool2D()(x)
    x = Conv2D(32, kernel_size=(1,1), padding='same', strides=1, activation='relu6')(x)

    x = TCN([32, 16, 16], kernel_size=3)(x)
    # x = logits = tf.reduce_mean(x, axis=2)
    x = Dense(NUM_CLASSES, kernel_initializer='he_normal', activation='softmax', name='softmax')(x)
    output = CTCLoss(name="ctc_loss")(labels, x)
    model = Model(inputs=[inputs, labels], outputs=output)
    return model


if __name__ == '__main__':
    inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL), batch_size=BATCH_SIZE, name='input_image')
    inputs = STN(name='STN')(inputs)
    model = TCN_LPR(inputs)
    model.summary()
