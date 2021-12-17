import sys
sys.path.append('./')

import matplotlib
matplotlib.use('Qt5Agg')

import glob
import os

import cv2
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config import *
from keras.preprocessing.image import ImageDataGenerator

# # img is in BGR format if the underlying image is a color image
# img = cv2.imdecode(np.fromfile('测试目录/test.jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)

# show the different license plate in B, G, R single channel


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


def pick_channel(image, channel_name='G'):
    assert channel_name in list('BGR')
    channel_index = list('BGR').index(channel_name)
    return cv2.split(image)[channel_index]


class LicensePlateGen(tf.keras.utils.Sequence):
    def __init__(self, directory, label_dict, target_size=(64, 96), channel_name='G', batch_size=128, shuffle=True, aug=True, **kwargs):
        self.directory = directory
        self.char2num = keras.layers.StringLookup(
            vocabulary=list(label_dict), mask_token=None
        )
        self.image_arr = glob.glob(os.path.join(directory, '*.jpg'))
        self.n = len(self.image_arr)
        self.channel_name = channel_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path):
        img = cv2.imdecode(np.fromfile(
            path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img = pick_channel(img, self.channel_name)
        # if self.aug:
        # img = random_augment(img)
        return img/255.

    def __get_output(self, label):
        # print(label)
        label = os.path.splitext(os.path.basename(label))[0]
        decode_dict = self.char2num(
            tf.strings.unicode_split(label, input_encoding="UTF-8"))
        return decode_dict

    def __get_data(self, batches):
        # path_batch = batches
        # label_batch = batches
        X_batch = np.asarray([self.__get_input(x) for x in batches])
        y_batch = np.asarray([self.__get_output(y) for y in batches])
        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.image_arr[index *
                                 self.batch_size:(index + 1) * self.batch_size]
        print(batches)
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size


if __name__ == '__main__':
    # show_images_channel(channel_name='color')
    # show_images_channel(channel_name='R')
    # show_images_channel(channel_name='G')
    # show_images_channel(channel_name='B')

    generator = LicensePlateGen(
        directory='images',
        label_dict=CHARS,
        target_size=(HEIGHT, WIDTH),
        channel_name='G',
        batch_size=BATCH_SIZE,
    )
    for i in range(0, len(generator)):
        x, y = generator[i]
        print('%s => %s' % (x, y))
        # print(x.shape, y.shape)
        break
    pass
