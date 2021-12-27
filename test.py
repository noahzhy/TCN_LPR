import random
import numpy as np
import keras.backend as K
import tensorflow as tf
import keras
import cv2

from models.model import TCN_LPR
import numpy as np
import keras.backend as K
import tensorflow as tf
import sys
sys.path.append('./')
from keras import layers
from config import *

from utils.utils import *
random.seed(SEED)


f = open('test_error.csv', 'w+', encoding="utf-8")
f.write("correct,error\n")

if __name__ == '__main__':
    path = TEST_DIR
    paths = glob.glob(os.path.join(path, '*.jpg'))
    sample_path = random.sample(paths, N_SAMPLE)
    counter = 0

    saved_model_dir = "model_tf_9700"
    model = keras.models.load_model(
        saved_model_dir,
        custom_objects={'<lambda>': lambda y_true, y_pred: y_pred}
    )
    model = keras.models.Model(
        model.get_layer(name="image").input,
        model.get_layer(name="softmax0").output
    )

    for img_path in sample_path:
        img, label = image_preprocess(img_path)
        img = tf.reshape(img, [-1, HEIGHT, WIDTH, CHANNEL])

        preds = model.predict(img)
        decoded_label = decode_label(preds)
        if decoded_label == label:
            counter += 1
        else:
            txt = "{},{}\n".format(label, decoded_label)
            f.write(txt)

    print('=> val acc: {}'.format(round((counter/N_SAMPLE)*100, 4)))
