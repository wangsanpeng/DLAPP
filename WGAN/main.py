# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from WGAN import *

log_dir = ''
ckpt_dir = ''
generated_sample_dir = ''
maxiter = 20000
n_critic = 5
#num_sample = 10

def train_wgan():
    dataset = input_data.read_data_sets('/Users/cheng/Data/mnist_data', one_hot=False)

    wgan = WGAN()
    critic_optimizer, generator_optimizer, critic_loss, generator_loss, z, real_data = wgan.build_graph()
#    z_sample, generated_samples = wgan.sample(num_sample)

    def get_next_bacth():
        train_data = dataset.train.next_batch(batch_size)[0]
        train_data = 2*train_data - 1
        train_data = np.reshape(train_data, (-1, 28, 28))
        npad = ((0, 0), (2, 2), (2, 2))
        train_data = np.pad(train_data, npad, mode='constant', constant_values=-1)
        train_data = np.expand_dims(train_data, -1)
        batch_z = np.random.normal(0, 1, (batch_size, z_dim)).astype(np.float32)
        feed_dict = {real_data: train_data, z: batch_z}
        return feed_dict

    merge_all = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)

        for iter in xrange(maxiter):
            if iter < 25 or iter % 500 == 0:
               N_Critic = 100
            else:
               N_Critic = n_critic

            for i_critic in xrange(N_Critic):
                feed_dict = get_next_bacth()
                if iter%100 == 50 and i_critic == 0:
                    _, c_loss, c_summary = sess.run([critic_optimizer, critic_loss, merge_all], feed_dict=feed_dict)
                    summary_writer.add_summary(c_summary, iter)
                    print "iter {}, iter_critic {}, critic_loss {}".format(iter, i_critic, c_loss)
                else:
                    _, c_loss = sess.run([critic_optimizer, critic_loss], feed_dict=feed_dict)
                    print "iter {}, iter_critic {}, critic_loss {}".format(iter, i_critic, c_loss)

            feed_dict = get_next_bacth()
            if iter % 100 == 99:
                _, g_loss, g_summary = sess.run([generator_optimizer, generator_loss, merge_all], feed_dict=feed_dict)
                summary_writer.add_summary(g_summary, iter)
                print "iter {}, generator_loss {}".format(iter, g_loss)
            else:
                _, g_loss = sess.run([generator_optimizer, generator_loss], feed_dict=feed_dict)
                print "iter {}, generator_loss {}".format(iter, g_loss)

#            if iter % 1000 == 0:
#                z_ = np.random.normal(0,1 (num_sample, z_dim)).astype(np.float32)
#                generated_samples_ = sess.run([generated_samples], feed_dict={z_sample: z_})

            if iter % 5000 == 0:
                saver.save(sess, os.path.join(ckpt_dir, 'model.ckpt'), global_step=iter)

if __name__ == '__main__':
    train_wgan()



