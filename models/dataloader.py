import sys
sys.path.append('./')

import matplotlib
matplotlib.use('Qt5Agg')

import random
import glob
import os

import cv2
import keras
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import tensorflow as tf
import matplotlib.pyplot as plt
from config import *
from keras.preprocessing.image import ImageDataGenerator
from utils.utils import *


class LicensePlateGen(tf.keras.utils.Sequence):
    def __init__(self, directory, label_dict, target_size=(64, 96), channel_name='G', batch_size=128, shuffle=False, aug=True, **kwargs):
        self.directory = directory
        self.target_size = target_size
        self.image_arr = glob.glob(os.path.join(directory, '*.jpg'))
        self.n = len(self.image_arr)
        self.channel_name = channel_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_channels = 1
        self.aug = aug
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.n / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        batches = self.image_arr[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(batches)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_arr))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batches):
        # Initialization
        X = np.empty((self.batch_size, *self.target_size, self.n_channels))
        y = np.full((self.batch_size, 8), NUM_CLASS, dtype=int)
        # Generate data
        for i, image_path in enumerate(batches):
            img, label = image_preprocess(image_path)

            # if self.aug:
            # img = random_augment(img)

            X[i] = img
            y[i][:len(encoded_label)] = encode_label(label, CHARS_DICT)

        return [X, y], y


if __name__ == '__main__':
    # show_images_channel(channel_name='color')
    # show_images_channel(channel_name='R')
    # show_images_channel(channel_name='G')
    # show_images_channel(channel_name='B')

    generator = LicensePlateGen(
        directory=r'images',
        label_dict=CHARS,
        target_size=(HEIGHT, WIDTH),
        channel_name='G',
        batch_size=BATCH_SIZE,
    )
    for i in range(0, len(generator)):
        x, y = generator[i]
        # print('%s, %s => %s' % (x['input_image'].shape, x['label'].shape, y.shape))
        # print(x.shape, y.shape)
        print(y)
        break
