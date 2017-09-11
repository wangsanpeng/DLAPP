#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/9/6 下午3:06
# @Author  : shaoguang.csg
# @File    : main.py

from models.read_mnist_data import read_mnist_data
from models.dense_net import DenseNet
from models.conf import ModelConf
from models.utils import logger

import tensorflow as tf
import numpy as np

import os
import time


def train(model_conf:ModelConf, train_set, validation_set=None):
    """
    train model for fashion mnist
    :param model_conf:
    :param train_set:
    :param validation_set:
    :return:
    """
    model_conf.print_conf()

    images_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, model_conf.HEIGHT*model_conf.WIDTH])
    labels_placeholder = tf.placeholder(dtype=tf.int64, shape=[None])
    model = DenseNet(model_conf, is_training=True, images=images_placeholder, labels=labels_placeholder)
    model.build_graph()

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_conf.SUMMARY_DIR, graph=sess.graph)

        for step in range(model_conf.NUM_STEPS):
            train_images, train_labels = train_set.next_batch()
            feed_dict = {images_placeholder: train_images,
                         labels_placeholder: train_labels
                         }
            start_time = time.time()
            [loss_value, acc_value, _, lr_value] = sess.run([model.loss, model.acc, model.train_op, model.learning_rate],
                                                            feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 10 == 0:
                _print_log('training', step, loss_value, acc_value, lr_value, duration)

            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            if step % 500 == 0:
                validation_images = validation_set.images
                validation_labels = validation_set.labels
                feed_dict = {
                    images_placeholder: validation_images,
                    labels_placeholder: validation_labels
                }
                [loss_value, acc_value] = sess.run([model.loss, model.acc], feed_dict=feed_dict)
                logger.info('(validation)loss: %f, acc: %f' % (loss_value, acc_value))

            if (step != 0 and step % 10000 == 0) or step + 1 == model_conf.NUM_STEPS:
                checkpoint_path = os.path.join(model_conf.MODEL_SAVER_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, step)


def test(model_conf: ModelConf, test_set):
    """
    test the performance of trained model
    :param model_conf:
    :param test_set:
    :return:
    """
    images_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, model_conf.HEIGHT*model_conf.WIDTH])
    labels_placeholder = tf.placeholder(dtype=tf.int64, shape=[model_conf.BATCH_SIZE])
    model = DenseNet(model_conf, is_training=False, images=images_placeholder, labels=labels_placeholder)
    model.build_graph()

    saver = tf.train.Saver()
    sess = tf.Session()
    try:
        ckpt_state = tf.train.get_checkpoint_state(model_conf.MODEL_SAVER_DIR)
    except tf.errors.OutOfRangeError as e:
        logger.info('Can not restore checkpoint %s', e)
        return

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        logger.info('No model to eval yet at %s', model_conf.MODEL_SAVER_DIR)
        return

    logger.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    num_step = int(test_set.num_examples/model_conf.BATCH_SIZE)+1
    num_correct = 0
    for step in range(num_step):
        test_images, test_labels = test_set.next_batch()
        feed_dict = {images_placeholder: test_images, labels_placeholder:test_labels}
        [loss, prediction, acc] = sess.run([model.loss, model.prediction, model.acc], feed_dict=feed_dict)
        num_correct += np.sum(np.argmax(prediction, axis=1) == test_labels)
        logger.info("(Test)batch %d, loss=%f, acc=%f" % (step, loss, acc))

    precision = num_correct*1.0/(num_step*model_conf.BATCH_SIZE)
    logger.info("Total precision: %f" % precision)


def _print_log(prefix, step, loss_value, acc_value, lr_value, duration):
    format_str = '(%s): step %d, loss = %.5f, acc = %.5f, lr = %.5f (%.3f examples/sec; %.3f sec/batch)'
    logger.info(format_str % (prefix, step, loss_value, acc_value, lr_value,
                        model_conf.BATCH_SIZE / duration, duration))


if __name__ == '__main__':
    model_conf = ModelConf()

    datasets = read_mnist_data(model_conf, validation_size=-1)
    train_set = datasets.train
    validation_set = datasets.validation
    test_set = datasets.test

    if model_conf.MODE == 'TRAIN':
        train(model_conf, train_set, validation_set)
    elif model_conf.MODE == 'TEST':
        test(model_conf, test_set)
    else:
        logger.info("Unsupported mode")