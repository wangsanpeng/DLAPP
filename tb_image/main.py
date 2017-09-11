#!/usr/bin/env python
# coding=utf-8

import os
from DLAPP.tb_image.make_train_test import split_train_test
from DLAPP.tb_image.create_tf_records import create_tf_records
from DLAPP.tb_image.basic_param import (
    TRAIN_FILE,
    TEST_FILE,
    VALIDATION_FILE
)

if __name__ == '__main__':
    # organize flow here
    # whether to split train and test
    if not (os.path.exists(TRAIN_FILE) and
                os.path.exists(TEST_FILE) and
                os.path.exists(VALIDATION_FILE)):
        split_train_test()

    if not (os.path.exists(TRAIN_FILE + '.tfrecord') and
                os.path.exists(TEST_FILE + '.tfrecord') and
                os.path.exists(VALIDATION_FILE + '.tfrecord')):
        create_tf_records()

