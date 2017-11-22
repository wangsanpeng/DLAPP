#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/9/11 下午8:44
# @Author  : shaoguang.csg
# @File    : model_conf.py

import yaml
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_conf_file', 'conf/model_conf.yaml', 'Path to the model config yaml file')

class ModelConf(object):
    def __init__(self):
        with open(FLAGS.model_conf_file) as model_conf_file:
            model_conf = yaml.load(model_conf_file)

        self.MODE = model_conf.get('MODE', '')
        self.TRAIN_TF_RECORD_PATH = model_conf.get('TRAIN_TF_RECORD_PATH', '')
        self.VALID_TF_RECORD_PATH = model_conf.get('VALID_TF_RECORD_PATH', '')
        self.SUMMARY_DIR = model_conf.get('SUMMARY_DIR', '')
        self.MODEL_SAVER_DIR = model_conf.get('MODEL_SAVER_DIR', '')

        self.HEIGHT = model_conf.get('HEIGHT', 224)
        self.WIDTH = model_conf.get('WIDTH', 224)
        self.NUM_CLASS = model_conf.get('NUM_CLASS', 80)

        self.BOTTLENECK = model_conf.get('BOTTLENECK', 'False')
        self.NUM_STEPS = model_conf.get('NUM_STEPS', 80000)
        self.VALID_NUM_STEPS = model_conf.get('VALID_NUM_STEPS', 100)
        self.BATCH_SIZE = model_conf.get('BATCH_SIZE', 128)
        self.DEPTH = model_conf.get('DEPTH', 13)
        self.GROWTH_RATE = model_conf.get('GROWTH_RATE', 32)
        self.COMPRESSION_TARE = model_conf.get('COMPRESSION_TARE', 0.5)
        self.WEIGHT_DECAY_RATE = model_conf.get('WEIGHT_DECAY_RATE', 0.0001)
        self.LEARNING_RATE = model_conf.get('LEARNING_RATE', 0.1)
        self.LR_DECAY_STEPS = model_conf.get('LR_DECAY_STEPS', 20000)

        self.DEBUG = bool(model_conf.get('DEBUG', 'True'))

        self.CUDA_DEVICE = model_conf.get('CUDA_DEVICE', 0)