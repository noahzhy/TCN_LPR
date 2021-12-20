import cv2
import os
import glob
import sys
sys.path.append('./')
from itertools import groupby
import numpy as np
import keras

from config import *


def encode_label(label, char_dict):
    encode = [char_dict[c] for c in label]
    return np.array(encode)


# show the different license plate in B, G, R single channel
def pick_channel(image, channel_name='G'):
    assert channel_name in list('BGR')
    channel_index = list('BGR').index(channel_name)
    return cv2.split(image)[channel_index]


def show_images_channel(channel_name):
    paths = glob.glob(os.path.join('images', '*.jpg'))

    plt.figure()
    for i in range(0, len(paths)):
        img = cv2.imread(paths[i])
        if channel_name.upper() == 'COLOR':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = pick_channel(img, channel_name)

        plt.subplot(3, 3, i+1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.suptitle('Channel Name {}'.format(channel_name))
    plt.show()


def image_preprocess(image_path, channel_name='G'):
    img = cv2.imdecode(np.fromfile(
        image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    img = pick_channel(img, channel_name)
    img = np.expand_dims(img, axis=-1)

    label = os.path.splitext(os.path.basename(image_path))[0].split('_')[0]
    return img/255., label


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
