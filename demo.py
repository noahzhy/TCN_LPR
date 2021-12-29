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
import time

from utils.utils import *
random.seed(SEED)


if __name__ == '__main__':
    path = r'C:\dataset\license_plate\license_plate_recognition\DatasetId_271674_1640677605\cut'
    paths = glob.glob(os.path.join(path, '*.jpg'))
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

    for img_path in paths:
        img, label = image_preprocess(img_path)
        img = tf.reshape(img, [-1, HEIGHT, WIDTH, CHANNEL])

        preds = model.predict(img)
        decoded_label = decode_label(preds)

        os.rename(img_path, os.path.join(path, "{}_{}.jpg".format(decoded_label, int(time.time()*1000))))
        time.sleep(0.001)
        
