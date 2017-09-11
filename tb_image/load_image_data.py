#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

from DLAPP.tb_image.utils import logger
from DLAPP.tb_image.basic_param import (
    TRAIN_FILE,
    TEST_FILE,
    VALIDATION_FILE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    BATCH_SIZE,
    DEBUG
)

def _read_and_decode(filename:str, is_augmentation:bool=False) -> tuple:
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, record_str = reader.read(filename_queue)
    example = tf.parse_single_example(record_str,
                                      features={
                                          'label': tf.FixedLenFeature([], tf.int64),
                                          'image_raw': tf.FixedLenFeature([], tf.string)
                                      })
    # parse image and label from example
    image = tf.decode_raw(example['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(example['label'], tf.int32)
    image = tf.reshape(image, shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3])

#    image = image/255.0
#    image = tf.image.per_image_standardization(image)

    return image, label


def get_batch_data(mode: str, num_threads:int=16)->tuple:
    """
    create image batch data for train/test/validation
    :param mode: candidate options: (train, test, validation)
    :param num_threads
    :return:
    """
    filename = ''
    if "train" == mode:
        filename = TRAIN_FILE
    elif "test" == mode:
        filename = TEST_FILE
    elif "validation" == mode:
        filename = VALIDATION_FILE
    else:
        logger.error("Only support [train/test/validation] mode")

    example_list = [_read_and_decode(filename+'.tfrecord') for _ in range(num_threads)]
#    example_list = [_read_and_decode(filename+'.tfrecord')]

    # generate batch for training
    min_after_dequeue = 5
    capacity = min_after_dequeue + 3 * BATCH_SIZE

    image_batch, label_batch = tf.train.shuffle_batch_join(example_list,
                                                           batch_size=BATCH_SIZE,
                                                           capacity=capacity,
                                                           min_after_dequeue=min_after_dequeue)
    if DEBUG:
        tf.summary.image("image_batch", image_batch, max_outputs=10)

    return image_batch, label_batch

if __name__ == '__main__':
    import numpy as np
#    import matplotlib.pyplot as plt

    image_batch, label_batch = get_batch_data(mode='train', num_threads=2)
    image_batch_test, label_batch_test = get_batch_data(mode='test', num_threads=2)
    init_op = tf.global_variables_initializer()
    merge_all = tf.summary.merge_all()

    session = tf.Session()
    session.run(init_op)

    summary_writer = tf.summary.FileWriter(logdir='/Users/cheng/tmp', graph=session.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    image, label, summary = session.run([image_batch, label_batch, merge_all])
    image_test, label_test, summary_test = session.run([image_batch_test, label_batch_test, merge_all])

    summary_writer.add_summary(summary_test, 1)
    print('label_test: '+ str(label_test))
    print(image_test[0].size)

    coord.request_stop()
    coord.join(threads)
    session.close()