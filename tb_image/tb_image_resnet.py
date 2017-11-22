#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_utils, resnet_v2

from DLAPP.tb_image.load_image_data import get_batch_data
from DLAPP.tb_image.basic_param import (CLASS_NAME)
