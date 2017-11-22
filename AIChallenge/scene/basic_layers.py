#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/9/5 下午3:06
# @Author  : shaoguang.csg
# @File    : basic_layers.py

import tensorflow as tf
import tensorflow.contrib.layers as layers


def batch_norm(x, is_training=False, decay=0.999, epsilon=1e-4, name='batch_norm'):
    x_shape = x.get_shape()
    param_shape = x_shape[-1]
    from tensorflow.python.training import moving_averages

    with tf.variable_scope(name) as scope:
        beta = tf.get_variable('beta', shape=param_shape, initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.0), trainable=is_training)
        gamma = tf.get_variable('gama', shape=param_shape, initializer=tf.random_normal_initializer(mean=1.0,stddev=0.002), trainable=is_training)

        global_mean = tf.get_variable('global_mean', shape=param_shape, initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.0),
                                      trainable=False)
        global_var = tf.get_variable('global_var', shape=param_shape, initializer=tf.constant_initializer(1.0),
                                     trainable=False)

        axis = list(range(len(x_shape)-1))
        mean,var = tf.nn.moments(x, axis)

        update_global_mean = moving_averages.assign_moving_average(global_mean, mean, decay)
        update_global_var = moving_averages.assign_moving_average(global_var, var, decay)

        def mean_var_update():
            with tf.control_dependencies([update_global_mean, update_global_var]):
                return tf.identity(mean), tf.identity(var)

        if is_training:
            mean, var = mean_var_update()
            return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
        else:
            return tf.nn.batch_normalization(x, global_mean, global_var, beta, gamma, epsilon)


def conv2d(x, output_channel, kernel_size, stride=1, name='conv2d'):
    with tf.variable_scope(name) as scope:
        return layers.conv2d(x, num_outputs=output_channel, kernel_size=kernel_size, stride=stride, padding='SAME',
                             activation_fn=tf.identity,
                             reuse=True, scope=scope,
                             weights_initializer=tf.orthogonal_initializer())


def max_pool2d(x, kernel_size, name='max_pool2d'):
    with tf.name_scope(name) as scope:
        return layers.max_pool2d(x, kernel_size=kernel_size, stride=2, padding='VALID')


def avg_pool2d(x, kernel_size, name='avg_pool2d'):
    with tf.name_scope(name) as scope:
        return layers.avg_pool2d(x, kernel_size=kernel_size, stride=2, padding='VALID')


def fully_connected(x, num_outputs, activation_fn, name='fully_connected'):
    with tf.variable_scope(name) as scope:
        return layers.fully_connected(x, num_outputs=num_outputs, activation_fn=activation_fn,
                                      reuse=True, scope=scope,
                                      weights_initializer=tf.orthogonal_initializer())


# def batch_norm(x, is_training, name='batch_norm'):
#    with tf.variable_scope(name) as scope:
#        return layers.batch_norm(x, is_training=is_training, center=True, scale=True, reuse=True, scope=scope)
