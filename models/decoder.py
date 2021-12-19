import numpy as np
import keras.backend as K
import tensorflow as tf


char2num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)


num2char = layers.StringLookup(
    vocabulary=char2num.get_vocabulary(), mask_token=None, invert=True
)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # greedy search
    results = K.ctc_decode(pred, input_length=input_len, greedy=True)
    results = results[0][0][:, :max_length]

    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num2char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


if __name__ == '__main__':
    pass