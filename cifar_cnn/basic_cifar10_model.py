# encoding: utf-8

"""
builder cnn model for cifar-10 dataset
"""

import tensorflow as tf
from basic_param import *

def weight_variable(name, shape):
    # using xavier initialize method
    initial = tf.truncated_normal(shape, mean=0.0,stddev=0.01)
    return tf.get_variable(name, initializer=initial, dtype=tf.float32)


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

def inference(images, evalution=False):
    if evalution:
        reuse = True
    else:
        reuse = False

    with tf.variable_scope('conv1', reuse=reuse) as scope:
 #        W_conv1 = weight_variable(scope.name+'/weights', [5, 5, 3, 64])
        W_conv1 = tf.get_variable(scope.name + '/weights',
                                  initializer=tf.truncated_normal([5, 5, 3, 64], mean=0.0, stddev=0.01),
                                  dtype=tf.float32)
        b_conv1 = bias_variable(scope.name + '/bias', [64])
        h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1, name=scope.name)
        h_pool1 = max_pool_2x2(h_conv1)
        regularizer_W1 = tf.nn.l2_loss(W_conv1, name='conv1_w_l2loss')
        tf.add_to_collection("l2loss", regularizer_W1)
        _collect_summery(h_pool1)

    norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2', reuse=reuse) as scope:
        W_conv2 = weight_variable(scope.name + '/weights', [5, 5, 64, 64])
        b_conv2 = bias_variable(scope.name + '/bias', [64])
        h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2, name=scope.name)
        h_pool2 = max_pool_2x2(h_conv2)
        regularizer_W2 = tf.nn.l2_loss(W_conv2, name='conv2_w_l2loss')
        tf.add_to_collection("l2loss", regularizer_W2)
        _collect_summery(h_pool2)

    norm2 = tf.nn.lrn(h_pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    with tf.variable_scope('conv3', reuse=reuse) as scope:
        W_conv3 = weight_variable(scope.name + '/weights', [3, 3, 64, 64])
        b_conv3 = bias_variable(scope.name + '/bias', [64])
        h_conv3 = tf.nn.relu(conv2d(norm2, W_conv3) + b_conv3, name=scope.name)
        h_pool3 = max_pool_2x2(h_conv3)
        regularizer_W3 = tf.nn.l2_loss(W_conv3, name='conv3_w_l2loss')
        tf.add_to_collection("l2loss", regularizer_W3)
        _collect_summery(h_pool3)

    with tf.variable_scope('fc1', reuse=reuse) as scope:
        reshape = tf.reshape(h_pool3, [FLAGS.batch_size, -1])
        W_fc1 = weight_variable(scope.name+'/weights', [reshape.get_shape()[1].value, 384])
        b_fc1 = bias_variable(scope.name+'/bias', [384])
        h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1)+b_fc1, name=scope.name)
        if evalution:
            keep_prob=1.0
        else:
            keep_prob=0.5
        h_dropout1 = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
        regularizer_fc1 = tf.nn.l2_loss(W_fc1, name='fc1_w_l2loss')
        tf.add_to_collection("l2loss", regularizer_fc1)
        _collect_summery(h_dropout1)

    with tf.variable_scope('fc2', reuse=reuse) as scope:
        W_fc2 = weight_variable(scope.name+'/weights', [384, 192])
        b_fc2 = bias_variable(scope.name+'/bias', [192])
        h_fc2 = tf.nn.relu(tf.matmul(h_dropout1, W_fc2)+b_fc2, name=scope.name)
        regularizer_fc2 = tf.nn.l2_loss(W_fc2, name='fc2_w_l2loss')
        tf.add_to_collection("l2loss", regularizer_fc2)
        _collect_summery(h_fc2)

    with tf.variable_scope('output', reuse=reuse) as scope:
        W_fc3 = weight_variable(scope.name+'/weights', [192, NUM_CLASS])
        b_fc3 = bias_variable(scope.name+'/bias', [NUM_CLASS])
        y_predict = tf.matmul(h_fc2, W_fc3)+b_fc3
        _collect_summery(y_predict)

    return y_predict


def loss(y_true, y_predict):
    # y_true非one-hot
    loss_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_predict, y_true), name='cross_entropy_loss')
    loss_ += 1e-4*sum(tf.get_collection('l2loss'))
    return loss_

def accuarcy(y_true, y_predict):
    return tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_true, tf.int64), tf.arg_max(y_predict, 1)), tf.float32), name='accuracy')

def solver(loss, global_step=None):
    lr = tf.train.exponential_decay(learning_rate=0.001,
                                    global_step=global_step,
                                    decay_steps=LR_DECAY_STEP,
                                    decay_rate=0.95,
                                    staircase=True)
    tf.scalar_summary('learning rate', lr)
    return tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step), lr



