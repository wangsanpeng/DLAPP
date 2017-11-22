#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/9/11 下午8:52
# @Author  : shaoguang.csg
# @File    : read_scene_data.py

from utils.logger import logger

from os import path
import json
from PIL import Image
import tensorflow as tf

SCENE_CLASS_FILE = "scene_classes.csv"

HEIGHT = 224
WIDTH = 224

TRAIN_PATH = "/Users/cheng/Data/AIChanllenge/scene/ai_challenger_scene_train_20170904"
VALID_PATH = "/Users/cheng/Data/AIChanllenge/scene/ai_challenger_scene_validation_20170908"
SCENE_TRAIN_LABEL_FILE = "scene_train_annotations_20170904_processed.json"
SCENE_VALID_LABEL_FILE = "scene_validation_annotations_20170908_processed.json"


def load_class_info(pathname):
    """

    :param pathname:
    :return:
    """
    scene_classes_path = path.join(pathname, SCENE_CLASS_FILE)
    with open(scene_classes_path) as fobj:
        lines = fobj.readlines()

    class_info = {}
    for line in lines:
        line_array = line.split(',')
        class_info[line_array[0]] = line_array[1]

    return class_info


def load_image_info(pathname):
    """
    load image informations from file
    :param pathname:
    :return:
    """
    with open(pathname) as fobj:
        lines = fobj.readlines()

    images_info = {}
    for line in lines:
        json_obj = json.loads(line.strip())
        images_info[json_obj["image_id"]] = json_obj["label_id"]

    return images_info


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tf_record(image_pathname, tf_record_name, images_info):
    """

    :param image_pathname:
    :param tf_record_name:
    :param images_info:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(tf_record_name)
    for image_id in images_info:
        filename = path.join(image_pathname, image_id)
        label = images_info[image_id]
        image = Image.open(filename).resize([HEIGHT, WIDTH], resample=Image.BILINEAR)
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(label)),
            'image_raw': _bytes_feature(image.tobytes())
        }))
        writer.write(example.SerializeToString())
    writer.close()


def _read_and_decode(filename, is_augmentation=False) :
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
    image = tf.reshape(image, shape=[HEIGHT, WIDTH, 3])
    label = tf.cast(example['label'], tf.int64)

    if is_augmentation:
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)

    return image, label


def show_tf_record(tf_record_name):
    """

    :param tf_record_name:
    :return:
    """
    if not path.exists(tf_record_name):
        logger.error("%s not exists." % tf_record_name)

    filename_queue = tf.train.string_input_producer([tf_record_name])
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)
    features = tf.parse_single_example(example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
#    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [HEIGHT, WIDTH, 3])
    label = tf.cast(features['label'], tf.int32)

#    image = tf.image.random_flip_left_right(image)
#    image = tf.image.random_brightness(image, max_delta=63)
#    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

    with tf.Session() as sess:
        import matplotlib.pyplot as plt

        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(5):
            example, l = sess.run([image, label])  # 在会话中取出image和label
#            img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
#            img.show()
            plt.imshow(example)
            plt.show()
        coord.request_stop()
        coord.join(threads)


def create_train_and_test_tf_record():
    train_images_info = load_image_info(path.join(TRAIN_PATH, SCENE_TRAIN_LABEL_FILE))
    valid_images_info = load_image_info(path.join(VALID_PATH, SCENE_VALID_LABEL_FILE))

    logger.info("train images: %d" % len(train_images_info))
    logger.info("valid images: %d" % len(valid_images_info))

    train_image_pathname = path.join(TRAIN_PATH, "scene_train_images_20170904")
    valid_image_pathname = path.join(VALID_PATH, "scene_validation_images_20170908")
    train_tf_record_name = path.join(TRAIN_PATH, "scene_train_images_20170904.tfrecord")
    valid_tf_record_name = path.join(VALID_PATH, "scene_validation_images_20170908.tfrecord")

    logger.info("Start to create tfrecord for train images: ")
    create_tf_record(train_image_pathname, train_tf_record_name, train_images_info)

    logger.info("Start to create tfrecord for valid images: ")
    create_tf_record(valid_image_pathname, valid_tf_record_name, valid_images_info)


if __name__ == '__main__':
    tf_record_name = path.join(TRAIN_PATH, "scene_train_images_20170904.tfrecord")
    show_tf_record(tf_record_name)
