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

# # img is in BGR format if the underlying image is a color image
# img = cv2.imdecode(np.fromfile('测试目录/test.jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)

# show the different license plate in B, G, R single channel

def encode_label(label, char_dict):
    # print(label)
    encode = [char_dict[c] for c in label]
    return encode


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
    def __init__(self, directory, label_dict, target_size=(64, 96), channel_name='G', batch_size=128, shuffle=False, aug=True, **kwargs):
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

    # def on_epoch_end(self):
    #     if self.shuffle:
    #         random.shuffle(self.image_arr)

    def __get_input(self, path):
        img = cv2.imdecode(np.fromfile(
            path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img = pick_channel(img, self.channel_name)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = np.expand_dims(img, axis=-1)
        # if self.aug:
        # img = random_augment(img)

        label = os.path.splitext(os.path.basename(path))[0].split('_')[0]
        encoded_label = encode_label(label, CHARS_DICT)

        return img/255., encoded_label

    # def __get_output(self, label):
    #     # print(label)
    #     label = os.path.splitext(os.path.basename(label))[0].split('_')[0]
    #     encoded_label = encode_label(label, CHARS_DICT)
    #     # decode_dict = self.char2num(
    #     #     tf.strings.unicode_split(label, input_encoding="UTF-8")
    #     # )
    #     # decode_dict = np.expand_dims(decode_dict, axis=-1)
    #     return encoded_label

    def __get_data(self, batches):
        x_list = []
        y_list = []
        for xy in batches:
            x, y = self.__get_input(xy)
            x_list.append(x)
            y_list.append(y)


        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)

        # X_batch = np.asarray([self.__get_input(x) for x in batches])
        # y_batch = np.asarray([self.__get_output(y) for y in batches])
        # return X_batch, y_batch
        return {"input_image": x_list, "label": y_list}, y_list
        # return [X_batch, y_batch], None

    def __getitem__(self, index):
        batches = self.image_arr[
            index *self.batch_size:(index + 1) * self.batch_size
        ]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size


class LicensePlateGen_v2(tf.keras.utils.Sequence):
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
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batches = self.image_arr[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # batches_temp = [self.image_arr[k] for k in batches]

        # Generate data
        X, y = self.__data_generation(batches)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_arr))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batches_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.target_size, self.n_channels))
        y = np.empty((self.batch_size, ), dtype=int)

        # Generate data
        for i, image_path in enumerate(batches_temp):
            img = cv2.imdecode(np.fromfile(
                image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            img = pick_channel(img, self.channel_name)
            img = cv2.resize(img, (WIDTH, HEIGHT))
            img = np.expand_dims(img, axis=-1)
            # if self.aug:
            # img = random_augment(img)

            label = os.path.splitext(os.path.basename(image_path))[0].split('_')[0]
            encoded_label = encode_label(label, CHARS_DICT)

            X[i] = img/255.
            y[i] = np.array(encoded_label)

        return X, y


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
        print('%s, %s => %s' % (x['input_image'].shape, x['label'].shape, y.shape))
        # print(x[0].get_shape(), y.get_shape())
        break
    pass
