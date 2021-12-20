import keras
from keras.callbacks import Callback



def calculate_acc(y_true, y_pred):
    counter = 0
    print(y_true.shape)
    total = y_true.shape[1]
    for batch in zip(preds, labels):
        if decode_label(batch[0]) == batch[1]:
            counter += 1

    return round(counter/total, 8)



class mCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        


if __name__ == '__main__':
    pass