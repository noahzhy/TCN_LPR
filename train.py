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
from models.cosine import *
from models.mcallback import *
from models.dataloader import *


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()


trainGen = LicensePlateGen(
    directory=TRAIN_DIR,
    # directory=r'images',
    label_dict=CHARS,
    target_size=(HEIGHT, WIDTH),
    channel_name='G',
    batch_size=BATCH_SIZE,
)

valGen = LicensePlateGen(
    directory=VAL_DIR,
    label_dict=CHARS,
    target_size=(HEIGHT, WIDTH),
    channel_name='G',
    batch_size=BATCH_SIZE,
)

warmup_batches = WARMUP_EPOCH * 112385 / BATCH_SIZE
total_steps = int(NUM_EPOCHS * 112385 / BATCH_SIZE)
# Compute the number of warmup batches.
warmup_steps = int(WARMUP_EPOCH * 112385 / BATCH_SIZE)

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=4e-05,
    warmup_steps=warmup_steps,
    hold_base_rate_steps=5,
)


def train(model, train_data, val_data):
    model.compile(
        loss=lambda y_true, y_pred: y_pred,
        optimizer=Adam(learning_rate=LEARNING_RATE),
        # run_eagerly=True
    )
    callbacks_list = [
        ModelCheckpoint(
            filepath='model_tf',
            monitor='val_loss',
            save_best_only=True,
        ),
        EarlyStopping(
            monitor='val_loss', 
            patience=30, 
            verbose=0, 
            mode='auto'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='auto',
            factor=0.5,
            patience=15,
        ),
        TensorBoard(log_dir='./logs'),
        warm_up_lr,
    ]
    model.fit(
        train_data,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=callbacks_list,
        validation_data=val_data,
        # validation_freq=10,
    )


if __name__ == '__main__':
    # inputs = Input(
    #     shape=(HEIGHT, WIDTH, CHANNEL),
    #     batch_size=BATCH_SIZE,
    #     name='stn_input',
    #     dtype="float32"
    # )
    # x = STN()(inputs)
    model = TCN_LPR()
    model.summary()
    train(model, trainGen, valGen)
