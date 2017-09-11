#!/usr/bin/env python
# coding=utf-8

import _io
import numpy as np

from DLAPP.tb_image.utils import (
    type_check,
    logger
)
from DLAPP.tb_image.basic_param import (
    URL_FILE,
    CLASS_NAME,
    CLASS_NAME_MAP,
    SPLIT_RATIO,
    TRAIN_FILE, VALIDATION_FILE, TEST_FILE
)


def _check_record_ok(line):
    """
    check whether line is a valid record
    :param line: split record
    :return:
    """
    if 3 != len(line):
        logger.debug("Broken record: %s", line)
        return False

    if line[1] is None or len(line[1]) == 0:
        logger.debug("Broken url: %s", line)
        return False

    return True


@type_check
def _write_data_to_file(data: list, file: _io.TextIOWrapper)->None:
    """
    write data to file
    :param data:
    :param file:
    :return:
    """
    file.writelines(data)
    file.flush()


def split_train_test():
    """
    split train, validation and test part for given dataset
    :return:
    """
    with open(URL_FILE, "r") as f:
        lines = f.readlines()

    # init class index map
    class_index_map = {}
    for class_name in CLASS_NAME:
        class_index_map[class_name] = []

    for index in range(len(lines)):
        line = lines[index].strip().split(',')
        if not _check_record_ok(line):
            continue

        title, url, class_id = line[0], line[1], CLASS_NAME_MAP[line[2]]
        class_index_map[class_id].append(index)

    [logger.info("Class id: %s, item: %d", k, len(v)) for k, v in class_index_map.items()]

    lines = np.array(lines)
    train_file = open(TRAIN_FILE, "w")
    validation_file = open(VALIDATION_FILE, "w")
    test_file = open(TEST_FILE, "w")
    for class_id in class_index_map.keys():
        image_list = class_index_map[class_id]
        np.random.shuffle(image_list) # shuffle image list
        train_num = int(len(image_list)*SPLIT_RATIO['train'])
        validation_num = int(len(image_list)*SPLIT_RATIO['validation'])
        test_num = len(image_list) - train_num - validation_num

        _write_data_to_file(lines[image_list[:train_num]].tolist(), train_file)
        _write_data_to_file(lines[image_list[train_num+1:train_num+validation_num]].tolist(), validation_file)
        _write_data_to_file(lines[image_list[train_num+validation_num+1:]].tolist(), test_file)

        logger.info("Class %s, train: %d, validation: %d, test: %d", class_id, train_num, validation_num, test_num)

    train_file.close()
    validation_file.close()
    test_file.close()


if __name__ == '__main__':
    split_train_test()