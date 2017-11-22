#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/9/18 PM3:52
# @Author  : shaoguang.csg
# @File    : resnet.py

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import get_or_create_global_step

from utils.logger import logger
from utils.utils import print_tensor_shape
from scene.basic_layers import (conv2d, avg_pool2d, max_pool2d, fully_connected, batch_norm)


class ResNet(object):
    """
    build residual net graph
    """
    def __init__(self, images, labels, model_conf, is_training):
        self._images = images
        self._labels = labels
        self.model_conf = model_conf
        self._is_training = is_training
        self.global_step = get_or_create_global_step()

        self._filters = [64, 64, 128, 256, 512]
        self._kernels = [7, 3, 3, 3, 3]
        self._stride = [2, 1, 2, 2, 2]

        if self._is_training:
            self._mode = "TRAIN"
            self._reuse = False
        else:
            self._mode = "Not TRAIN"
            self._reuse = True

        logger.info("In %s phase" % (self._mode))

    def _residual_block_A(self, x, out_channel, stride, activation_fn=tf.nn.relu, name="residual_block_A"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            if in_channel == out_channel:
                if stride == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = max_pool2d(x, kernel_size=3)
            else:
                shortcut = conv2d(x, output_channel=out_channel, kernel_size=1, stride=stride)

            # residual
            x = conv2d(x, output_channel=out_channel, kernel_size=3, stride=stride, name='conv2d_1')
            x = batch_norm(x, is_training=self._is_training, name='bn_1')
            x = activation_fn(x)
            x = conv2d(x, output_channel=out_channel, kernel_size=3, name='conv2d_2')
            x = batch_norm(x, is_training=self._is_training, name='bn_2')

            # merge
            x = x + shortcut
            x = activation_fn(x)
        return x

    def _residual_block_B(self, x, activation_fn=tf.nn.relu, name="residual_block_B"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            shortcut = x
            x = conv2d(x, output_channel=num_channel, kernel_size=3, name='conv2d_1')
            x = batch_norm(x, is_training=self._is_training, name='bn_1')
            x = activation_fn(x)
            x = conv2d(x, output_channel=num_channel, kernel_size=3, name='conv2d_2')
            x = batch_norm(x, is_training=self._is_training, name='bn_2')

            # merge
            x = x + shortcut
            x = activation_fn(x)
        return x

    def resnet(self, x, activation_fn=tf.nn.relu):
        with tf.variable_scope('resnet_conv1', reuse=self._reuse) as scope:
            x = conv2d(x, output_channel=self._filters[0], kernel_size=self._kernels[0], stride=self._stride[0])
            x = batch_norm(x, self._is_training)
            x = activation_fn(x)
            x = max_pool2d(x, kernel_size=3)  # [n,64,56,56]

        with tf.variable_scope('resnet_block1', reuse=self._reuse) as scope:
            x = self._residual_block_B(x, name='block_B_1')
            x = self._residual_block_B(x, name='block_B_2') # [n,64,56,56]
            print_tensor_shape(x, "After block_B_2")

        with tf.variable_scope('resnet_block2', reuse=self._reuse) as scope:
            x = self._residual_block_A(x, out_channel=self._filters[2], stride=self._stride[2], name='block_A_1')
            x = self._residual_block_B(x, name='block_B_3') # [n,128,28,28]
            print_tensor_shape(x, "After block_B_3")

        with tf.variable_scope('resnet_block3', reuse=self._reuse) as scope:
            x = self._residual_block_A(x, out_channel=self._filters[3], stride=self._stride[3], name='block_A_2')
            x = self._residual_block_B(x, name='block_B_4') # [n,256,14,14]
            print_tensor_shape(x, "After block_B_4")

        with tf.variable_scope('resnet_block4', reuse=self._reuse) as scope:
            x = self._residual_block_A(x, out_channel=self._filters[4], stride=self._stride[4], name='block_A_3')
            x = self._residual_block_B(x, name='block_B_5') # [n,512,7,7]
            print_tensor_shape(x, "After block_B_5")

        with tf.variable_scope('global_pooling') as scope:
            height = x.get_shape().as_list()[1]
            x = avg_pool2d(x, kernel_size=height)
        print_tensor_shape(x, "After average pooling")

        with tf.variable_scope('resnet_end', reuse=self._reuse) as scope:
            x = tf.reshape(x, shape=[-1, x.get_shape().as_list()[-1]])
            logit = fully_connected(x, self.model_conf.NUM_CLASS, activation_fn=tf.identity)

        return logit

    def build_graph(self):
        logit = self.resnet(self._images)
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=self._labels)
        cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')

        # cal l2loss
        l2_costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'weight') > 0:
                l2_costs.append(tf.nn.l2_loss(var))
                tf.summary.histogram(var.op.name, var)

#        [l2_costs.append(tf.nn.l2_loss(var)) for var in tf.trainable_variables() if var.op.name.find(r'weight') > 0]

        self.l2loss = self.model_conf.WEIGHT_DECAY_RATE*tf.add_n(l2_costs)
        self.loss = cross_entropy_loss + self.model_conf.WEIGHT_DECAY_RATE*tf.add_n(l2_costs)
        self.prediction = layers.softmax(logit)
        _labels = tf.arg_max(self.prediction, 1)
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(_labels, self._labels)))

        if self._is_training:
            self.build_train_op()

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('l2loss', self.l2loss)
        tf.summary.scalar('accuracy', self.acc)
        tf.summary.image('image', self._images, max_outputs=10)

    def build_train_op(self):
        self.learning_rate = tf.train.exponential_decay(learning_rate=self.model_conf.LEARNING_RATE,
                                                        global_step=self.global_step,
                                                        decay_steps=self.model_conf.LR_DECAY_STEPS,
                                                        decay_rate=0.1,
                                                        staircase=True)

        self.learning_rate = tf.maximum(self.learning_rate, 1e-5)
        tf.summary.scalar('learning_rate', self.learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
        self.train_op= optimizer.minimize(self.loss, self.global_step, name='train_step')
