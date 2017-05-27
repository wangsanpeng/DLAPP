# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as tl

def leak_relu(x, leak=0.2, name='lrelu'):
    """
    y = max(x, leak*x)
    :param x:
    :param leak:
    :param name:
    :return:
    """
    with tf.name_scope(name):
        return tf.maximum(x, leak*x)

image_size = 32
num_channel = 1

batch_size = 64
z_dim = 128
learning_rate_critic = 0.00005
learning_rate_generator = 0.00005

class WGAN(object):

    def __init__(self, ):
        pass

    def build_graph(self):
        z = tf.placeholder(dtype=tf.float32, shape=(batch_size, z_dim))
        generator = self._generator_conv
        critic = self._critic_conv
        fake_data = generator(z)
        real_data = tf.placeholder(dtype=tf.float32, shape=(batch_size, image_size, image_size, num_channel))

        # compute loss
        fake_logit = critic(fake_data)
        real_logit = critic(real_data, reuse=True)
        critic_loss = tf.reduce_mean(fake_logit-real_logit)
        generator_loss = -tf.reduce_mean(fake_logit)

        # create global step
        critic_global_step = tf.Variable(initial_value=0, trainable=False, name='critic_step')
        generator_global_step = tf.Variable(initial_value=0, trainable=False, name='generator_step')

        # collect trainable variable for critic and generator
        critic_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        generator_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # create optimizer
        critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_critic)\
            .minimize(critic_loss, global_step=critic_global_step, var_list=critic_var)
        generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_generator)\
            .minimize(generator_loss, global_step=generator_global_step, var_list=generator_var)

        clipped_critc_var = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in critic_var]

        with tf.control_dependencies([critic_optimizer]):
            critic_optimizer = tf.tuple(clipped_critc_var)

        tf.summary.scalar('critic_loss', critic_loss)
        tf.summary.scalar('generator_loss', generator_loss)
        tf.summary.image('generated_samples', fake_data, max_outputs=10)

        return critic_optimizer, generator_optimizer, critic_loss, generator_loss, z, real_data

    def sample(self, n_sample):
        z = tf.placeholder(dtype=tf.float32, shape=(n_sample, z_dim))
        generator = self._generator_conv
        generated_samples = generator(z)
        generated_samples = tf.div(tf.add(generated_samples, 1.0), 2.0)
        return z, generated_samples

    def _generator_conv(self, z):
        with tf.variable_scope('generator') as scope:
            net = tl.fully_connected(z, 4*4*256, activation_fn=leak_relu, normalizer_fn=tl.batch_norm)
            net = tf.reshape(net, shape=(-1,4,4,256))
            net = tl.conv2d_transpose(net, num_outputs=128,
                                      kernel_size=3,
                                      padding='SAME',
                                      stride=2,
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=tl.batch_norm,
                                      weights_initializer=tf.random_normal_initializer(0, 0.02))
            net = tl.conv2d_transpose(net, num_outputs=64,
                                      kernel_size=3,
                                      padding='SAME',
                                      stride=2,
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=tl.batch_norm,
                                      weights_initializer=tf.random_normal_initializer(0, 0.02))
            net = tl.conv2d_transpose(net, num_outputs=32,
                                      kernel_size=3,
                                      padding='SAME',
                                      stride=2,
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=tl.batch_norm,
                                      weights_initializer=tf.random_normal_initializer(0, 0.02))
            net = tl.conv2d_transpose(net, num_outputs=num_channel,
                                      kernel_size=3,
                                      padding='SAME',
                                      stride=1,
                                      activation_fn=tf.nn.tanh,
                                      weights_initializer=tf.random_normal_initializer(0, 0.02))
        return net

    def _critic_conv(self, image, reuse=False):
        with tf.variable_scope('critic') as scope:
            if reuse:
                scope.reuse_variables()
            size = 32
            image = tl.conv2d(image, num_outputs=size,
                              kernel_size=3,
                              stride=2,
                              activation_fn=leak_relu)
            image = tl.conv2d(image, num_outputs=size * 2,
                              kernel_size=3,
                              stride=2,
                              activation_fn=leak_relu,
                              normalizer_fn=tl.batch_norm)
            image = tl.conv2d(image, num_outputs=size * 4, kernel_size=3,
                              stride=2,
                              activation_fn=leak_relu,
                              normalizer_fn=tl.batch_norm)
            image = tl.conv2d(image, num_outputs=size * 8,
                              kernel_size=3,
                              stride=2,
                              activation_fn=leak_relu,
                              normalizer_fn=tl.batch_norm)
            logit = tl.fully_connected(tf.reshape(
                image, [batch_size, -1]), 1, activation_fn=None)
        return logit

