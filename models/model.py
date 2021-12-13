from tensorflow.keras import *
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
import tensorflow as tf
import keras

from models.tcn import TCN
from config import *


class CTCLoss(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
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


def TCN_LPR():
    inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL), batch_size=BATCH_SIZE, name='input_image')
    labels = Input(shape=(None,), batch_size=BATCH_SIZE, name="label")

    x = ConvBlock(32, kernel_size=(3,3), strides=(2,2), padding='same')(inputs)
    x = TCN([64, 64, 64, 128], kernel_size=5)(x)
    # logits = tf.reduce_mean(x, axis=2)
    # x = logits
    x = Dense(NUM_CLASSES, kernel_initializer='he_normal', activation='softmax', name='softmax')(x)
    output = CTCLoss(name="ctc_loss")(labels, x)
    model = Model(inputs=[inputs, labels], outputs=output)
    return model


if __name__ == '__main__':
    model = TCN_LPR()
    model.summary()
