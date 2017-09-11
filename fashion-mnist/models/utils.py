#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/9/6 下午3:06
# @Author  : shaoguang.csg
# @File    : utils.py

import logging


def _create_logger(log_level, log_format="", log_file=""):
    if log_format == "":
        log_format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'

    logger = logging.getLogger("")
    logger.setLevel(log_level)
#    logging.getLogger('requests.packages.urllib3.connectionpool').setLevel(logging.WARN)

    formatter = logging.Formatter(log_format)
    if log_file != "":
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = _create_logger(logging.INFO)