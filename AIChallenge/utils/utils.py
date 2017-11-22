#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/9/18 PM5:29
# @Author  : shaoguang.csg
# @File    : utils.py

import tensorflow as tf
from utils.logger import logger


def print_tensor_shape(x:tf.Variable, prefix=""):
    shape = x.get_shape().as_list()
    logger.info(prefix + ", shape: %s" % str(shape))
