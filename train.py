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


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()


# trainGen = ThermalDataGen(
#     'train_data.csv',
#     batch_size=BATCH_SIZE,
#     num_frames=NUM_FRAMES,
#     shuffle=True,
#     aug=True,
# )
# valGen = ThermalDataGen(
#     'val_data.csv',
#     num_frames=NUM_FRAMES,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     aug=False,
# )


def train(model, train_data, val_data):
    callbacks_list = [
        ModelCheckpoint(
            filepath='model_tf',
            monitor='val_loss',
            save_best_only=True,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='auto',
            factor=0.1,
            patience=10,
        )
    ]
    model.summary()
    model.compile(
        metrics=['accuracy'],
        optimizer=Adam(learning_rate=LEARNING_RATE),
        # loss='categorical_crossentropy',
    )
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
    model.summary()
    # train(model, trainGen, valGen)
