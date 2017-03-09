
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

seed = 42

np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

class DataDistribution(object):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, N, verbose=False):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()

        if verbose:
            plt.hist(samples, 100)
            plt.show()

        return samples

class GeneratorDistribution(object):
    """
    self-generated data distribution
    """
    def __init__(self, range):
        self.range = range

    def sample(self, N):
#        return np.random.uniform(-self.range, self.range, N)
        return np.linspace(-self.range, self.range, N)  + np.random.random(N)*0.01


def linear(input, out_dim, scope=None, stddev=1.0):
    W_init = tf.random_normal_initializer(stddev=stddev)
    b_init = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        W = tf.get_variable("W", [input.get_shape()[1], out_dim],  initializer=W_init)
        b = tf.get_variable("b", [out_dim], initializer=b_init)
        return tf.matmul(input, W)+b

def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, 8, 'g0'))
    h1 = tf.nn.softplus(linear(h0, 8, 'g1'))
    h2 = linear(h1, 1, 'g2')
    return h2

def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.tanh(linear(input, 8, 'd0'))
    h1 = tf.tanh(linear(h0, 8, 'd1'))
    h2 = tf.tanh(linear(h1, 10, 'd2'))

    if minibatch_layer:
        h3 = minibatch(h2)
    else:
        h3 = tf.tanh(linear(h2, 10, 'd3'))

    h4 = tf.sigmoid(linear(h3, 1, 'd4'))

    return h4

def minibatch(input, num_kernels=5, num_dims=3):
    x = linear(input, num_kernels * num_dims, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, num_dims))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat(1, [input, minibatch_features])

def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.95
    num_decay_steps = 100
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer