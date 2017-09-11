#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import os
import collections
from PIL import Image

from DLAPP.tb_image.make_train_test import _check_record_ok
from DLAPP.tb_image.basic_param import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_PATH,
    TRAIN_FILE, VALIDATION_FILE, TEST_FILE,
    CLASS_NAME_MAP
)
from DLAPP.tb_image.utils import logger

Dataset = collections.namedtuple("Dataset", {'images_path', 'labels', 'text'})


def _parse_file(filename:str) -> Dataset:
    """
    parse images and labels from filename
    :param filename:
    :return:
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    dataset = Dataset(images_path=[], labels=[], text=[])
    for line in lines:
        line = line.strip().split(',')
        if not _check_record_ok(line):
            continue
        dataset.images_path.append(line[1])
        dataset.labels.append(CLASS_NAME_MAP[line[2]].strip())
        dataset.text.append(line[0])

    return dataset


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _write_dataset_to_tfrecord(dataset: Dataset, filename:str) -> None:
    """
    write dataset to tfrecord file
    :param dataset:
    :param filename:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(filename)

    num = len(dataset.labels)
    for index in range(num):
        label = dataset.labels[index]
        image_name = dataset.images_path[index]
        image_path = os.path.join(IMAGE_PATH, label, image_name)
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path) .resize([IMAGE_HEIGHT, IMAGE_WIDTH], resample=Image.BILINEAR)

        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(label)),
            'image_raw': _bytes_feature(image.tobytes())
        }))
        writer.write(example.SerializeToString())
    writer.close()


def create_tf_records():
    """
    create tfrecords for train/test/validation
    :return:
    """
    filenames = [TRAIN_FILE, VALIDATION_FILE, TEST_FILE]
    for filename in filenames:
        logger.info("Write %s to tfrecord", filename)
        dataset = _parse_file(filename)
        record_file = filename + '.tfrecord'
        _write_dataset_to_tfrecord(dataset, record_file)


if __name__ == "__main__":
    create_tf_records()