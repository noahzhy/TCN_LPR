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


def calculate_acc(y_true, y_pred):
    counter = 0
    print(y_true.shape)
    total = y_true.shape[1]
    for batch in zip(y_true, y_pred):
        if batch[0] == decode_label(batch[1]):
            counter += 1

    return round(counter/total, 8)



trainGen = LicensePlateGen(
    directory=r'C:\dataset\license_plate\license_plate_recognition\train',
    # directory=r'images',
    label_dict=CHARS,
    target_size=(HEIGHT, WIDTH),
    channel_name='G',
    batch_size=BATCH_SIZE,
)

valGen = LicensePlateGen(
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
        EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            verbose=0, 
            mode='auto'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='auto',
            factor=0.1,
            # patience=10,
        ),
        TensorBoard(log_dir='./logs'),
    ]
    model.compile(
        loss=lambda y_true, y_pred: y_pred,
        optimizer=Adam(learning_rate=LEARNING_RATE),
        # metrics=['accuracy', calculate_acc]
        # run_eagerly=True
    )
    model.fit(
        train_data,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=callbacks_list,
        validation_data=val_data,
        validation_freq=10,
    )


if __name__ == '__main__':
    model = TCN_LPR()
    # model.load_weights('model.h5')
    model.summary()
    train(model, trainGen, valGen)
