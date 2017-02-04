'''
An example of distribution approximation using Generative Adversarial Networks in TensorFlow.
The minibatch discrimination technique is taken from Tim Salimans et. al.: https://arxiv.org/abs/1606.03498.
'''

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

class GAN(object):
    """A simple demo to show how gan works"""

    def __init__(self, data, gen, num_steps, d_steps, batch_size, is_minibatch, log_every, save_fig=False):
        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.is_minibatch = is_minibatch
        self.g_hid_dim = 3
        self.d_hid_dim = 6
        self.d_steps = d_steps
        self.log_every = log_every
        self.save_fig = save_fig

        if self.is_minibatch:
            self.learning_rate = 0.003
        else:
            self.learning_rate = 0.01

        self._create_model()

    def _create_model(self):
        # pretraining discriminator
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.pre_label = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.d_hid_dim, self.is_minibatch)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre-self.pre_label))
            self.pre_optimizer = optimizer(self.pre_loss, None, self.learning_rate)

        # generator
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.G = generator(self.z, self.g_hid_dim)

        # discriminator
        with tf.variable_scope('D') as scope:
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.D1 = discriminator(self.x, self.d_hid_dim, self.is_minibatch)
            scope.reuse_variables()
            self.D2 = discriminator(self.G, self.d_hid_dim, self.is_minibatch)

        self.loss_d = tf.reduce_mean(-0.9*tf.log(self.D1)-0.1*tf.log(1-self.D1)-tf.log(1-self.D2)) # one-side label smoothing
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))

        vars = tf.trainable_variables()
        self.d_pre_params = [v for v in vars if v.name.startswith('D_pre/')]
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.d_optimizer = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.g_optimizer = optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        # pretraining discriminator
        num_pretrain_steps = 1000
        for step in xrange(num_pretrain_steps):
            d = (np.random.random(self.batch_size)-0.5)*10.0
            labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
            pre_loss,_ = sess.run([self.pre_loss, self.pre_optimizer],feed_dict={
                self.pre_input: np.reshape(d, (self.batch_size, 1)),
                self.pre_label: np.reshape(labels, (self.batch_size, 1))
            })

        self.d_weights = sess.run(self.d_pre_params)

        # copy weights from pretraining to discrimator network
        for index, value in enumerate(self.d_params):
            sess.run(value.assign(self.d_weights[index]))

        # training D and G
        for step in xrange(self.num_steps):
            # update D
            loss_d_ = 0
            for d_step in xrange(self.d_steps):
                x = self.data.sample(self.batch_size)
                z = self.gen.sample(self.batch_size)

                loss_d, _ = sess.run([self.loss_d, self.d_optimizer], feed_dict={
                    self.x: np.reshape(x, (self.batch_size, 1)),
                    self.z: np.reshape(z, (self.batch_size, 1))
                })
                loss_d_ = loss_d

            # update G
            z = self.gen.sample(self.batch_size)
            loss_g, _ = sess.run([self.loss_g, self.g_optimizer], feed_dict={
                self.z: np.reshape(z, (self.batch_size, 1))
            })

            if step % self.log_every == 0:
                print('{}: {}\t{}'.format(step, loss_d_, loss_g))

        self._plot_distribution(sess)


    def _samples(self, sess, num_points=20000, num_bins=100):
        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # decision boundary
        db = np.zeros((num_points, 1))
        for index in range(num_points//self.batch_size):
            db[self.batch_size*index:self.batch_size*(index+1)] = sess.run(self.D1, {
                self.x: np.reshape(
                    xs[self.batch_size*index:self.batch_size*(index+1)],
                    (self.batch_size, 1)
                )
            })

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # data generation
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for index in range(num_points//self.batch_size):
            g[self.batch_size*index:self.batch_size*(index+1)] = sess.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * index:self.batch_size * (index + 1)],
                    (self.batch_size, 1)
                )
            })

        pg,_ = np.histogram(g, bins=bins, density=True)

        return db, pd, pg

    def _plot_distribution(self, session):
        db, pd, pg = self._samples(session)
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        pd_x = np.linspace(-self.gen.range, self.gen.range, len(pd))

        plt.plot(db_x, db, label='boundary')
        plt.plot(pd_x, pd, label='real data distribution')
        plt.plot(pd_x, pg, label='generated data distribution')
        plt.xlabel('data values')
        plt.ylabel('density')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    model = GAN(
        DataDistribution(-1, 0.5),
        GeneratorDistribution(range=6),
        num_steps=1000,
        d_steps=2,
        batch_size=16,
        is_minibatch=True,
        log_every=10
    )

    model.train()