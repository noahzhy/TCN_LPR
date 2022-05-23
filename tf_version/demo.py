import random
import sys
sys.path.append('./')
import time
from PIL import Image
import cv2
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf

from config import *
from utils.utils import *

random.seed(SEED)


if __name__ == '__main__':
    path = 'D:\dataset\license_plate\double_lp'
    paths = glob.glob(os.path.join(path, '*.jpg'))
    counter = 0

    saved_model_dir = "tf_version/model_tf_9770"
    model = keras.models.load_model(
        saved_model_dir,
        custom_objects={'<lambda>': lambda y_true, y_pred: y_pred}
    )
    model = keras.models.Model(
        model.get_layer(name="image").input,
        model.get_layer(name="softmax0").output,
    )

    for img_path in paths:
        img, label = image_preprocess(img_path)
        img = tf.reshape(img, [-1, HEIGHT, WIDTH, CHANNEL])

        preds = model.predict(img)
        decoded_label = decode_label(preds)

        print(decoded_label, os.path.basename(img_path).split('.')[0])

        # os.rename(img_path, os.path.join(path, "{}_{}.jpg".format(decoded_label, int(time.time()*1000))))
        time.sleep(0.001)
