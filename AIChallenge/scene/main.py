#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/9/12 下午10:47
# @Author  : shaoguang.csg
# @File    : main.py

from utils.logger import logger
from scene.model_conf import ModelConf
from scene.read_scene_data import get_batch_data
from scene.dense_net import DenseNet
# from scene.resnet import ResNet

import tensorflow as tf
from time import time
import os


def train(model_conf:ModelConf):
    """
    :param model_conf:
    :param train_set:
    :param valid_set:
    :return:
    """
    train_images, train_labels = get_batch_data(model_conf.TRAIN_TF_RECORD_PATH, is_augmentation=True)
    train_model = DenseNet(model_conf=model_conf, is_training=True, images=train_images, labels=train_labels)
    train_model.build_graph()

    valid_images, valid_labels = get_batch_data(model_conf.TRAIN_TF_RECORD_PATH, is_augmentation=False)
    valid_model = DenseNet(model_conf=model_conf, is_training=False, images=valid_images, labels=valid_labels)
    valid_model.build_graph()

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        summary_writer = tf.summary.FileWriter(model_conf.SUMMARY_DIR, graph=sess.graph)

        for step in range(model_conf.NUM_STEPS):
            start_time = time()
            [loss_value, acc_value, _, lr_value] = sess.run([train_model.loss, train_model.acc, train_model.train_op, train_model.learning_rate])
            duration = time() - start_time
            if step % 10 == 0:
                _print_log('training', step, loss_value, acc_value, lr_value, duration, model_conf.BATCH_SIZE)

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 500 == 0:
                for valid_index in range(model_conf.VALID_NUM_STEPS):
                    [loss_value, acc_value] = sess.run([valid_model.loss, valid_model.acc])
                    logger.info('(validation)loss: %f, acc: %f' % (loss_value, acc_value))

            if (step != 0 and step % 10000 == 0) or step + 1 == model_conf.NUM_STEPS:
                checkpoint_path = os.path.join(model_conf.MODEL_SAVER_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, step)

        coord.request_stop()
        coord.join(threads)


def _print_log(prefix, step, loss_value, acc_value, lr_value, duration, batch_size):
    format_str = '(%s): step %d, loss = %.5f, acc = %.5f, lr = %.5f (%.3f examples/sec; %.3f sec/batch)'
    logger.info(format_str % (prefix, step, loss_value, acc_value, lr_value, batch_size / duration, duration))

if __name__ == '__main__':
    model_conf = ModelConf()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(model_conf.CUDA_DEVICE)

    if model_conf.MODE == 'TRAIN':
        train(model_conf)
    else:
        logger.info("Unsupported mode")

