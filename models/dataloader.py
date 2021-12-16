import cv2
import os
from PIL import Image
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import glob


# show the different license plate in B, G, R single channel
def show_images_channel(channel_name):
    paths = glob.glob(os.path.join('images', '*.jpg'))

    plt.figure()
    for i in range(0,len(paths)):
        img = cv2.imread(paths[i])
        if channel_name.upper() == 'COLOR':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = pick_channel(img, channel_name)

        plt.subplot(3,3,i+1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.suptitle('Channel Name {}'.format(channel_name))
    plt.show()



def pick_channel(image, channel_name='G'):
    assert channel_name in list('BGR')
    channel_index = list('BGR').index(channel_name)
    return cv2.split(image)[channel_index]


if __name__ == '__main__':
    show_images_channel(channel_name='color')
    show_images_channel(channel_name='R')
    show_images_channel(channel_name='G')
    show_images_channel(channel_name='B')
