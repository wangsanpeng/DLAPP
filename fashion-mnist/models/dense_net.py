#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/9/5 上午10:06
# @Author  : shaoguang.csg
# @File    : densenet.py

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import get_or_create_global_step

from models.utils import logger
from models.basic_layers import (conv2d, avg_pool2d, max_pool2d, fully_connected, batch_norm)


class DenseNet(object):
    """
    build dense net graph
    """
    def __init__(self, model_conf, is_training, images, labels):
        self._depth = model_conf.DEPTH
        self._growth_rate = model_conf.GROWTH_RATE
        self.model_conf = model_conf
        self._num_layer_per_block = int((self._depth-4)/3)

        self._images = tf.reshape(images, shape=[-1, self.model_conf.HEIGHT, self.model_conf.WIDTH, 1])
        self._labels = labels
        self._is_training = is_training
        self._layer_func = self.bottleneck_layer if self.model_conf.BOTTLENECK else self.add_layer
        self.global_step = get_or_create_global_step()

        logger.info("In %s phase, using %s as layer function" % (self.model_conf.MODE, self._layer_func.__name__))

    def add_layer(self, x, name, activation_fn=tf.nn.relu):
        with tf.variable_scope(name) as scope:
            xx = batch_norm(x, self._is_training)
            xx = activation_fn(xx)
            xx = conv2d(xx, self._growth_rate, kernel_size=3)
            x = tf.concat([xx, x], axis=3)
        return x

    def bottleneck_layer(self, x, name, activation_fn=tf.nn.relu):
        with tf.variable_scope(name) as scope:
            xx = batch_norm(x, self._is_training, name=name+'batch_norm_1')
            xx = activation_fn(xx)
            xx = conv2d(xx, 4*self._growth_rate, kernel_size=1, name=name+"conv2d_1")
            xx = batch_norm(xx, self._is_training, name=name+'batch_norm_2')
            xx = activation_fn(xx)
            xx = conv2d(xx, self._growth_rate, kernel_size=3, name=name+"conv2d_2")
            x = tf.concat([xx, x], axis=3)
        return x

    def add_transition(self, x, name, activation_fn=tf.nn.relu):
        input_channel = x.get_shape().as_list()[3]
        with tf.variable_scope(name) as scope:
            xx = batch_norm(x, self._is_training)
            xx = activation_fn(xx)
            xx = conv2d(xx, input_channel, kernel_size=1)
            xx = activation_fn(xx)
            xx = avg_pool2d(xx, kernel_size=2)
        return xx

    def dense_net(self, x, activation_fn=tf.nn.relu):
        with tf.variable_scope('densenet_conv1') as scope:
            x = conv2d(x, output_channel=16, kernel_size=3)

        with tf.variable_scope('dense_block_1') as scope:
            for index in range(self._num_layer_per_block):
                x = self._layer_func(x, name="layer_{}".format(index+1), activation_fn=activation_fn)
            x = self.add_transition(x, name="transition_1", activation_fn=activation_fn)

        if self.model_conf.DEBUG:
            logger.debug("After dense_block_1 the data shape is {}".format(str(x.get_shape().as_list())))

        with tf.variable_scope('dense_block_2') as scope:
            for index in range(self._num_layer_per_block):
                x = self._layer_func(x, name="layer_{}".format(index+1), activation_fn=activation_fn)
            x = self.add_transition(x, name="transition_2", activation_fn=activation_fn)

        if self.model_conf.DEBUG:
            logger.debug("After dense_block_2 the data shape is {}".format(str(x.get_shape().as_list())))

        with tf.variable_scope('dense_block_3') as scope:
            for index in range(self._num_layer_per_block):
                x = self.add_layer(x, name="layer_{}".format(index+1), activation_fn=activation_fn)

        with tf.variable_scope('global_pooling') as scope:
            x = batch_norm(x, self._is_training)
            x = activation_fn(x)
            height = x.get_shape().as_list()[1]
            x = avg_pool2d(x, kernel_size=height)

        if self.model_conf.DEBUG:
            logger.debug("After avg_pool2d the data shape is {}".format(str(x.get_shape().as_list())))

        with tf.variable_scope('dense_net_end') as scope:
            x = tf.reshape(x, shape=[-1, x.get_shape().as_list()[-1]])

            if self.model_conf.DEBUG:
                logger.debug("After squeeze the data shape is {}".format(str(x.get_shape().as_list())))
            logit = fully_connected(x, self.model_conf.NUM_CLASS, activation_fn=tf.identity)
        return logit

    def build_graph(self):
        logit = self.dense_net(self._images)
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=self._labels)
        cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')

        # cal l2loss
        l2_costs = []
        [l2_costs.append(tf.nn.l2_loss(var)) for var in tf.trainable_variables() if var.op.name.find(r'weight') > 0]

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

        self.learning_rate = tf.maximum(self.learning_rate, 1e-6)
        tf.summary.scalar('learning_rate', self.learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
        self.train_op= optimizer.minimize(self.loss, self.global_step, name='train_step')

