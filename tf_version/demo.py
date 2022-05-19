import random
import sys
sys.path.append('./')
import time

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

    saved_model_dir = "model_tf_9770"
    loaded = tf.saved_model.load(saved_model_dir)
    # tf.saved_model.save(model, "tf_model")
    print(list(loaded.signatures.keys()))  # ["serving_default"]
    infer = loaded.signatures["serving_default"]
    # tmpdir = tempfile.mkdtemp()
    # module_with_signature_path = os.path.join(tmpdir, 'module_with_signature')
    # call = module.__call__.get_concrete_function(tf.TensorSpec(None, tf.float32))
    # tf.saved_model.save(module, module_with_signature_path, signatures=call)
    print(infer.structured_outputs)
    print("load_from_tf", "done!")

    quit()
    model = keras.models.load_model(
        saved_model_dir,
        custom_objects={'<lambda>': lambda y_true, y_pred: y_pred}
    )
    model = keras.models.Model(
        model.get_layer(name="image").input,
        model.get_layer(name="softmax0").output,
    )
    model.save("tcn9770.h5", True, True)

    for img_path in paths:
        img, label = image_preprocess(img_path)
        img = tf.reshape(img, [-1, HEIGHT, WIDTH, CHANNEL])

        preds = model.predict(img)
        decoded_label = decode_label(preds)

        print(decoded_label, os.path.basename(img_path).split('.')[0])

        # os.rename(img_path, os.path.join(path, "{}_{}.jpg".format(decoded_label, int(time.time()*1000))))
        time.sleep(0.001)
