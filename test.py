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



if __name__ == '__main__':
    image_path = r'images/A42다3311.jpg'
    img = image_preprocess(image_path)[0]
    img = tf.reshape(img, [-1, HEIGHT, WIDTH, 1])

    saved_model_dir = "model_tf"
    model = keras.models.load_model(
        saved_model_dir,
        custom_objects={'<lambda>': lambda y_true, y_pred: y_pred}
    )
    model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="softmax0").output
    )
    # model.summary()
    # model.save('model.h5')

    print(img.shape)
    preds = model.predict(img)
    print(preds.shape)
    print('decode label:', decode_label(preds))
    quit()
