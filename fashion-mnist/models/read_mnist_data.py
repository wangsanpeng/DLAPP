#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/9/4 下午9:06
# @Author  : shaoguang.csg
# @File    : read_mnist_data.py

import os
import struct
import gzip
from collections import namedtuple

import numpy as np
from models.conf import ModelConf

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

Datasets = namedtuple("Datasets", ['train', 'test', 'validation'])


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        struct.unpack(">IIII", imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


class Dataset(object):
    def __init__(self, images, labels, batch_size, norm_type='min_max'):
        assert images.shape[0] == labels.shape[0], ""

        self._num_example = images.shape[0]
        self._images = images
        self._labels = labels
        self._current_pos = 0
        self._current_epoch = 0
        self._batch_size = batch_size

        if norm_type == 'min_max':
            self._images = self._images*1.0/255.0
        elif norm_type == 'std':
            self._images = (self._images*1.0-72.9403)/90.02118
        else:
            print("Unsupported norm_type")

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def current_epoch(self):
        return self._current_epoch

    @property
    def current_pos(self):
        return self._current_pos

    @property
    def num_examples(self):
        return self._num_example

    def next_batch(self):
        start = self._current_pos
        self._current_pos += self._batch_size

        if self._current_pos > self._num_example:
            self._current_pos = 0
            self._current_epoch += 1

            perm = np.arange(self._num_example)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            # start next epoch
            start = 0
            self._current_pos = self._batch_size
        end = self._current_pos

        return self._images[start:end], self._labels[start:end]


def read_mnist_data(model_conf, validation_size=5000):
    train_images, train_labels = load_mnist(model_conf.DATA_PATH, kind='train')
    test_images, test_labels = load_mnist(model_conf.DATA_PATH, kind='t10k')

    # shuffle the training images for split training and validation set
    perm = np.arange(train_images.shape[0])
    np.random.shuffle(perm)
    train_images = train_images[perm]
    train_labels = train_labels[perm]

    if validation_size > 0 :
        validation = Dataset(train_images[:validation_size], train_labels[:validation_size], model_conf.BATCH_SIZE)
    else:
        validation = Dataset(train_images, train_labels, model_conf.BATCH_SIZE)
    train = Dataset(train_images, train_labels, model_conf.BATCH_SIZE)
    test = Dataset(test_images, test_labels, model_conf.BATCH_SIZE)

    return Datasets(train=train, test=test, validation=validation)


if __name__ == '__main__':
    model_conf = ModelConf()
    datasets = read_mnist_data(model_conf)
    train_images, train_labels = datasets.train.next_batch()
    print(datasets.train.num_examples)
    print(datasets.test.num_examples)