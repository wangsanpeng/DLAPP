# encoding: utf-8

"""
builder FitNet-4 model for cifar-10 dataset
"""

from __future__ import division

import math
import tensorflow as tf
from basic_param import *


def _xaiver_initializer(n_inputs, n_outputs, uniform=True):
    """Set the parameter initialization using the method described.
      This method is designed to keep the scale of the gradients roughly the same
      in all layers.
      Xavier Glorot and Yoshua Bengio (2010):
               Understanding the difficulty of training deep feedforward neural
               networks. International conference on artificial intelligence and
               statistics.
      Args:
        n_inputs: The number of input nodes into each output.
        n_outputs: The number of output nodes for each input.
        uniform: If true use a uniform distribution, otherwise use a normal.
      Returns:
        An initializer.
      """
    if uniform:
        init = math.sqrt(6.0 / (n_inputs+n_outputs))
        return tf.random_uniform_initializer(-init, init)
    else:
        stddev = math.sqrt(3.0/(n_inputs+n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def _msra_initializer(n_inputs):
    """
    :param n_inputs:
    :return:
    """
    stddev = math.sqrt(2.0/n_inputs)
    return tf.truncated_normal_initializer(stddev=stddev)


def _max_out(x, nchannel_out, axis=None, name='max_out'):
    """
    max-out activiation function.
    From paper 'Maxout'
    :return:
    """
    if axis is None:
        axis = -1 # I just assume that channel is the last dimension
    shape = x.get_shape.as_list()
    n_channels = shape[-1]
    shape[-1] = axis
    shape += [n_channels//nchannel_out]
    output = tf.reduce_max(tf.reshape(x, shape), -1, keep_dims=False)
    return output


def weight_variable(name, shape, initial_type='xavier'):
    """
    :param name:
    :param shape:
    :param initial_type: 'xavier', 'msra', 'gassian', 'ortho'
    :return:
    """
    if initial_type == 'gassian':
        initial = tf.truncated_normal(shape, mean=0.0,stddev=0.01)
        return tf.get_variable(name, initializer=initial, dtype=tf.float32)
    elif initial_type == 'xaiver':
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    elif initial_type == 'msra':
        return _msra_initializer(shape[0])
    elif initial_type == 'ortho':
        pass
    else:
        pass


def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial, dtype=tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def _collect_summery(x):
    tf.histogram_summary(x.op.name + "_activations", x)
    tf.scalar_summary(x.op.name + "_sparsity", tf.nn.zero_fraction(x))

