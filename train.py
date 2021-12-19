import os
import glob
import numpy as np
import cv2

import keras
import keras.backend as K
import tensorflow as tf
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizer_v2.adam import *
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from config import *
from models.model import *
from models.dataloader import *


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()



# directory = r'C:\dataset\license_plate\license_plate_recognition\train'
# # directory = r'images'
# images = glob.glob(os.path.join(directory, '*.jpg'))
# labels = [os.path.splitext(os.path.basename(img))[0].split('_')[0] for img in images]
# print("Number of images found: ", len(images))
# print("Number of labels found: ", len(labels))

# def split_data(images, labels, train_size=0.8, shuffle=True):
#     # 1. Get the total size of the dataset
#     size = len(images)
#     # 2. Make an indices array and shuffle it, if required
#     indices = np.arange(size)
#     if shuffle:
#         np.random.shuffle(indices)
#     # 3. Get the size of training samples
#     train_samples = int(size * train_size)
#     # 4. Split data into training and validation sets
#     x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
#     x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
#     return x_train, x_valid, y_train, y_valid


# def encode_single_sample(img_path, label):
#     img = cv2.imdecode(
#         np.fromfile(img_path, dtype=np.uint8),
#         cv2.IMREAD_UNCHANGED
#     )
#     img = pick_channel(img, self.channel_name)
#     img = cv2.resize(img, (WIDTH, HEIGHT))
#     img = np.expand_dims(img, axis=-1)

#     encoded_label = encode_label(label, CHARS_DICT)
#     return {"input_image": img, "label": encoded_label}


# x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = (
#     train_dataset.map(
#         encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
#     )
#     .batch(BATCH_SIZE)
#     .prefetch(buffer_size=tf.data.AUTOTUNE)
# )

# validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
# validation_dataset = (
#     validation_dataset.map(
#         encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
#     )
#     .batch(BATCH_SIZE)
#     .prefetch(buffer_size=tf.data.AUTOTUNE)
# )

trainGen = LicensePlateGen_v2(
    directory=r'C:\dataset\license_plate\license_plate_recognition\train',
    # directory=r'images',
    label_dict=CHARS,
    target_size=(HEIGHT, WIDTH),
    channel_name='G',
    batch_size=BATCH_SIZE,
)

valGen = LicensePlateGen_v2(
    directory=r'C:\dataset\license_plate\license_plate_recognition\val',
    # directory=r'images',
    label_dict=CHARS,
    target_size=(HEIGHT, WIDTH),
    channel_name='G',
    batch_size=BATCH_SIZE,
)

def train(model, train_data, val_data):
    callbacks_list = [
        ModelCheckpoint(
            filepath='model_tf',
            monitor='val_loss',
            save_best_only=True,
        ),
        # ModelCheckpoint(
        #     filepath = 'model_{epoch:02d}_{val_loss:.2f}.h5'
        # ),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='auto',
            factor=0.1,
            patience=10,
        ),
        TensorBoard(log_dir='./logs'),
    ]
    model.summary()
    # model.compile(
    #     # metrics=['accuracy'],
    #     optimizer=Adam(learning_rate=LEARNING_RATE),
    #     # loss='sparse_categorical_crossentropy',
    #     # run_eagerly=True
    # )
    model.fit(
        train_data,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=callbacks_list,
        validation_data=val_data,
    )


if __name__ == '__main__':
    model = TCN_LPR()
    # model.load_weights('model.h5')
    # model.summary()
    train(model, trainGen, valGen)
    # model.save('model_tf')
