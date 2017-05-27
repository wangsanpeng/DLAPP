# -*- coding:utf-8 -*-
# @author shaoguang.csg

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import copy

from DLAPP.FacialKeypointDetection.conf import *


def load_data(is_test=False):
    """
    load face image and keypoints from csv file
    :param is_test:
    :return:
    """
    filename = TEST_FILE if is_test else TRAIN_FILE
    df = pd.read_csv(filename)

    df = df.dropna()
    df['Image'] = df['Image'].apply(lambda image: np.fromstring(image, sep=" ", count=SIZE*SIZE)/255.0)
    x = np.vstack(df['Image']).reshape((-1, SIZE, SIZE, N_CHANNEL))

    if is_test:
        y = None
    else:
        y = df[df.columns[:-1]].values/SIZE

    return x, y

def flip_images(images, keypoints, direction='horizon'):
    """
    flip images according to specified direction
    :param images: (n, w, h, c)
    :param keypoints: (n,)
    :param direction:
    :return:
    """
    keypoints_ = copy.deepcopy(keypoints)
    keypoints_ = keypoints_.reshape((keypoints.shape[0], -1, 2))
    if direction == 'horizonal':
        images_ = np.flipud(images)
        keypoints_[:, :, 0] = 1.0-keypoints_[:, :, 0]
        return images_, keypoints_.reshape((keypoints.shape[0], -1))
    elif direction == 'vertical':
        images_ = np.fliplr(images)
        keypoints_[:, :, 1] = 1.0-keypoints_[:, :, 1]
        return images_, keypoints_.reshape((keypoints.shape[0], -1))
    else:
        print("Unsupport direction")


def visualize_face(images, keypoints, is_scale=True):
    """
    visualize images and keypoints in face
    :param images: numpy.ndarray
    :param keypoints:
    :return:
    """
    assert len(images.shape) == 3 and \
           len(keypoints.shape) == 2 and \
           images.shape[0] == keypoints.shape[0]

    if is_scale:
        images = copy.deepcopy(images)
        keypoints = copy.deepcopy(keypoints)
        images *= 255
        keypoints *= SIZE

    fig = plt.figure()

    num_image = images.shape[0]
    num_col = int(np.ceil(np.sqrt(num_image)))
    num_row = int(np.ceil(num_image*1.0/num_col))
    keypoints = keypoints.reshape((num_image,-1,2))

    for row in range(num_row):
        for col in range(num_col):
            index = row*num_col+col
            if index >= num_image:
                break
            fig.add_subplot(num_row, num_col, index+1)
            plt.imshow(images[index,:,:], cmap='gray')
            plt.scatter(keypoints[index,:,0], keypoints[index,:,1], s=20, color='red')

if __name__ == '__main__':
    x, y = load_data()
    visualize_face(x[:10,:,:,0], y[:10,:], is_scale=True)
    x_, y_ = flip_images(x[:10,:,:,0], y[:10,:], direction='vertical')
    visualize_face(x_, y_, is_scale=True)
    print("")







