"""
Residual network for cifar-10 dataset (input size 32x32x3)
"""

import tensorflow as tf
import numpy as np

from DLAPP.DLBase.layers import *
from DLAPP.DLBase.activation import *
from DLAPP.DLBase.initializer import *

from functools import wraps


def lazy_property(func):
    attribute = '_cache_' + func.__name__

    @property
    @wraps(func)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return decorator

class ResNet(object):
    """residual net model"""

    def __init__(self, model_conf, images, labels):
        """
        :param model_conf:
        :param images:
        :param labels:
        """
        self.model_conf = model_conf
        self.images = images
        self.labels = labels
        self.mode = self.model_conf.mode

    def build_graph(self):
        from tensorflow.contrib.framework import get_or_create_global_step
        self.global_step = get_or_create_global_step()

        self._build_model()
        if self.mode == 'train':
            self._build_train_op()

    def _build_model(self):
        """
        build residual net
        :return:
        """
        x = InputLayer(inputs=self.images)
        x = Conv2dLayer(x, act_func=tf.identity,
                        shape=[3,3,3,16],
                        stride=[1,1,1,1],
                        W_init=msra_initializer(3 * 3 * 16),
                        name='init_conv')

        if self.model_conf.use_bottleneck:
            res_func = self._bottleneck_residual_cell
            filters = [16, 64, 128, 256]
        else:
            res_func = self._residual_cell
            filters = [16, 16, 32, 64]

        # unit 1
        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], stride=[1,1,1,1], activate_before_residual=True)

        for index in xrange(1, self.model_conf.num_residual_units):
            with tf.variable_scope('unit_1_%d' % index):
                x = res_func(x, filters[1], filters[1], stride=[1,1,1,1], activate_before_residual=False)

        # unit 2
        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], stride=[1, 2, 2, 1], activate_before_residual=False)

        for index in xrange(1, self.model_conf.num_residual_units):
            with tf.variable_scope('unit_2_%d' % index):
                x = res_func(x, filters[2], filters[2], stride=[1, 1, 1, 1], activate_before_residual=False)

        # unit 3
        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], stride=[1, 2, 2, 1], activate_before_residual=False)

        for index in xrange(1, self.model_conf.num_residual_units):
            with tf.variable_scope('unit_3_%d' % index):
                x = res_func(x, filters[3], filters[3], stride=[1, 1, 1, 1], activate_before_residual=False)

        # unit last
        with tf.variable_scope('unit_last'):
            x = BatchNormLayer(x, epsilon=0.001, is_train=self.mode, name='final_bn')
            x = LeakyReluLayer(x, self.model_conf.leakiness)

            assert x.outputs.get_shape().ndims == 4
            x.outputs = tf.reduce_mean(x.outputs, [1,2])

        x.outputs = tf.reshape(x.outputs, [self.model_conf.batch_size, -1])

        with tf.variable_scope('logit'):
            logit = DenseLayer(x, num_units=self.model_conf.num_classes,
                           act_func=tf.identity,
                           w_init=tf.uniform_unit_scaling_initializer(factor=1.0),
                           b_init=tf.constant_initializer())
            self.predictions = tf.nn.softmax(logit.outputs)

        self.network = logit

        with tf.variable_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logit.outputs, self.labels)
            self.loss = tf.reduce_mean(loss, name='cross_entropy_loss')

            costs = []
            for var in tf.trainable_variables():
                if var.op.name.find(r'W_') > 0:
                    costs.append(tf.nn.l2_loss(var))

            l2loss = self.model_conf.weight_decay_rate * tf.add_n(costs)
            self.loss += self.model_conf.weight_decay_rate * tf.add_n(costs)

            tf.summary.scalar('l2loss', l2loss)
            tf.summary.scalar('loss', self.loss)

    def _build_train_op(self):
        self.lrn_rate = tf.train.exponential_decay(learning_rate=self.model_conf.lrn_rate,
                                        global_step=self.global_step,
                                        decay_steps=self.model_conf.lr_decay_steps,
                                        decay_rate=0.1,
                                        staircase=True)

        tf.summary.scalar('learning rate', self.lrn_rate)

        if self.model_conf.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.model_conf.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)


        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step, name='train_step')


    def _residual_cell(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        """
        A residual cell with two sub-layer
        :param x:
        :param in_filter:
        :param out_filter:
        :param stride:
        :param activate_before_residual:
        :return:
        """
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = BatchNormLayer(x, epsilon=0.001, is_train=self.mode, name='init_bn')
                x = LeakyReluLayer(x, self.model_conf.leakiness, name='init_relu')
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = BatchNormLayer(x, epsilon=0.001, is_train=self.mode, name='init_bn')
                x = LeakyReluLayer(x, self.model_conf.leakiness, name='init_relu')

        with tf.variable_scope('sub1'):
            x = Conv2dLayer(x, act_func=tf.identity,
                            shape=[3,3,in_filter, out_filter],
                            W_init=msra_initializer(3*3*out_filter),
                            stride=stride,
                            name='conv1')

        with tf.variable_scope('sub2'):
            x = BatchNormLayer(x, epsilon=0.001, is_train=self.mode, name='bn2')
            x = LeakyReluLayer(x, self.model_conf.leakiness, name='relu2')
            x = Conv2dLayer(x, act_func=tf.identity,
                            shape=[3,3,out_filter,out_filter],
                            W_init=msra_initializer(3 * 3 * out_filter),
                            stride=[1,1,1,1],
                            name='conv2')

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x.outputs = tf.nn.avg_pool(orig_x.outputs, stride, stride, 'VALID')
                orig_x.outputs = tf.pad(orig_x.outputs, [[0,0],[0,0],[0,0],
                                          [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
            x.outputs += orig_x.outputs

            tf.logging.info('image after unit %s', x.outputs.get_shape())

            return x

    def _bottleneck_residual_cell(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        """
        bottleneck residual unit with 3 sub layer
        :param x:
        :param in_filter:
        :param out_filter:
        :param stride:
        :param activate_before_residual:
        :return:
        """
        if activate_before_residual:
            with tf.variable_scope('shared_activation') as scope:
                x = BatchNormLayer(x, epsilon=0.001, is_train=self.mode, name='init_bn')
                x = LeakyReluLayer(x, self.model_conf.leakiness, name='init_relu')
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation') as scope:
                orig_x = x
                x = BatchNormLayer(x, epsilon=0.001, is_train=self.mode, name='init_bn')
                x = LeakyReluLayer(x, self.model_conf.leakiness, name='init_relu')

        with tf.variable_scope('sub1') as scope:
            x = Conv2dLayer(x, act_func=tf.identity,
                            shape=[1,1,in_filter, out_filter/4],
                            W_init=msra_initializer(3*3*out_filter/4),
                            stride=stride,
                            name='conv1')

        with tf.variable_scope('sub2'):
            x = BatchNormLayer(x, epsilon=0.001, is_train=self.mode, name='bn2')
            x = LeakyReluLayer(x, self.model_conf.leakiness, name='relu2')
            x = Conv2dLayer(x, act_func=tf.identity,
                            shape=[3, 3, out_filter/4, out_filter/4],
                            W_init=msra_initializer(3 * 3 * out_filter/4),
                            stride=[1, 1, 1, 1],
                            name='conv2')

        with tf.variable_scope('sub3') as scope:
            x = BatchNormLayer(x, epsilon=0.001, is_train=self.mode, name='bn3')
            x = LeakyReluLayer(x, self.model_conf.leakiness, name='relu3')
            x = Conv2dLayer(x, act_func=tf.identity,
                            shape=[1, 1, out_filter/4, out_filter],
                            W_init=msra_initializer(3 * 3 * out_filter),
                            stride=[1, 1, 1, 1],
                            name='conv3')
        with tf.variable_scope('sub_add') as scope:
            if in_filter != out_filter:
                orig_x = Conv2dLayer(orig_x, act_func=tf.identity,
                                     shape=[1,1,in_filter, out_filter],
                                     W_init=msra_initializer(3 * 3 * out_filter),
                                     stride=stride)
            x.outputs += orig_x.outputs

        tf.logging.info('image after unit %s', x.outputs.get_shape())

        return x