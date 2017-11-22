#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/9/12 下午5:21
# @Author  : shaoguang.csg
# @File    : read_scene_data.py

from scene.model_conf import ModelConf
from utils.logger import logger
from scene.create_tf_record import _read_and_decode

import tensorflow as tf
from os import path


def get_batch_data(tf_record_filename, num_threads=16, batch_size=64, is_augmentation=False):
    """
    read data from batch
    :param mode:
    :param num_thread:
    :return:
    """
    if not path.exists(tf_record_filename):
        logger.error(tf_record_filename + " not exists")

    example_list = [_read_and_decode(tf_record_filename, is_augmentation=is_augmentation) for _ in range(num_threads)]
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch_join(example_list,
                                                           batch_size=batch_size,
                                                           capacity=capacity,
                                                           min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch