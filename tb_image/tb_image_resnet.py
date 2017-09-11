#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_utils, resnet_v2

from DLAPP.tb_image.load_image_data import get_batch_data
from DLAPP.tb_image.basic_param import (CLASS_NAME)


def train():
    global_step = tf.Variable(0, trainable=False)

    train_images_batch, train_label_batch = get_batch_data(mode='train')
    validation_images_batch, validation_label_batch = get_batch_data(mode='validation')

    bottleneck = resnet_v2.bottleneck
    blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(128, 32, 1)] * 2 + [(128, 32, 2)]),
        resnet_utils.Block('block2', bottleneck,
                           [(256, 64, 1)] * 3 + [(256, 64, 2)]),
        resnet_utils.Block('block3', bottleneck,
                           [(512, 128, 1)] * 5 + [(512, 128, 2)]),
        resnet_utils.Block('block4', bottleneck, [(1024, 256, 1)] * 3)
    ]

    logits, end_points = resnet_v2.resnet_v2(train_images_batch, blocks, num_classes=len(CLASS_NAME), reuse=True)
    training_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_label_batch, logits=logits))
    accuracy =
