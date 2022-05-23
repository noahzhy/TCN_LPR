import tensorflow as tf
import cv2
from PIL import Image
import os
import glob
import random
import numpy as np
from itertools import groupby


path = "model_uint8.tflite"
CHARS = "0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호바사아자배abcdefghijklmnopqABCDEFGHIJKLMNOPQ"  # exclude IO
# CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}


def decode_label(mat, chars=DECODE_DICT) -> str:
    # mat is the output of model
    # get char indices along best path
    best_path_indices = np.argmax(mat[0], axis=1)
    # print(best_path_indices)
    # collapse best path (using itertools.groupby), map to chars, join char list to string
    best_chars_collapsed = [chars[k]
                            for k, _ in groupby(best_path_indices) if k != len(chars)]
    res = ''.join(best_chars_collapsed)
    return res


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


# show the different license plate in B, G, R single channel
def pick_channel(image, channel_name='G'):
    assert channel_name in list('BGR')
    channel_index = list('BGR').index(channel_name)
    return cv2.split(image)[channel_index]


def image_preprocess(image_path, channel_name='G'):
    img = cv2.imdecode(np.fromfile(
        image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (96, 32))
    img = pick_channel(img, channel_name)
    img = np.expand_dims(img, axis=-1)
    return img


def demo(model_path, image_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print(input_details)
    output_details = interpreter.get_output_details()
    print(output_details)
    # quit()

    # img = Image.open(image_path).convert('L').resize((96, 32))
    img = image_preprocess(image_path)
    # # img.show()
    # imgs = np.zeros((1,32,96,1))
    # imgs[0, :, :, 0] = np.asarray(img, dtype=np.float32)
    imgs = tf.reshape(img, [-1, 32, 96, 1])
    # imgs = tf.cast(imgs, tf.float32)
    imgs = tf.cast(imgs, tf.uint8)

    # imgs = convert(imgs, 0, 255, np.uint8)
    # Image.fromarray(imgs[0, :, :, 0]).show()
    interpreter.set_tensor(input_details[0]['index'], imgs)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    res = decode_label(output_data)
    print(res)


if __name__ == "__main__":
    demo(path, "D:\mit_intern/2.jpg")
