#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/9/6 下午3:06
# @Author  : shaoguang.csg
# @File    : conf.py

import yaml
import tensorflow as tf

from models.utils import logger

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_conf_file', 'model_conf/model_conf.yaml', 'Path to the model config yaml file')


class ModelConf(object):
    def __init__(self):
        with open(FLAGS.model_conf_file) as model_conf_file:
            model_conf = yaml.load(model_conf_file)

        self.MODE = model_conf.get('MODE', '')

        self.DATA_PATH = model_conf.get('DATA_PATH', '')
        self.SUMMARY_DIR = model_conf.get('SUMMARY_DIR', '')
        self.MODEL_SAVER_DIR = model_conf.get('MODEL_SAVER_DIR', '')

        self.HEIGHT = model_conf.get('HEIGHT', 28)
        self.WIDTH = model_conf.get('WIDTH', 28)
        self.NUM_CLASS = model_conf.get('NUM_CLASS', 10)

        self.BOTTLENECK = model_conf.get('BOTTLENECK', 'False')
        self.NUM_STEPS = model_conf.get('NUM_STEPS', 60000)
        self.BATCH_SIZE = model_conf.get('BATCH_SIZE', 128)
        self.DEPTH = model_conf.get('DEPTH', 7)
        self.GROWTH_RATE = model_conf.get('GROWTH_RATE', 12)
        self.WEIGHT_DECAY_RATE = model_conf.get('WEIGHT_DECAY_RATE', 0.0001)
        self.LEARNING_RATE = model_conf.get('LEARNING_RATE', 0.1)
        self.LR_DECAY_STEPS = model_conf.get('LR_DECAY_STEPS', 20000)

        self.DEBUG = model_conf.get('DEBUG', 'True')

    def print_conf(self):
        logger.info("------------Model Conf----------------")
        logger.info("BOTTLENECK: %s" % str(self.BOTTLENECK))
        logger.info("NUM_STEPS : %d" % self.NUM_STEPS)
        logger.info("BATCH_SIZE: %d" % self.BATCH_SIZE)
        logger.info("DEPTH     : %d" % self.DEPTH)
        logger.info("GROWTH_RATE: %d" % self.GROWTH_RATE)
        logger.info("WEIGHT_DECAY_RATE: %f" % self.WEIGHT_DECAY_RATE)
        logger.info("LEARNING_RATE: %f" % self.LEARNING_RATE)
        logger.info("LR_DECAY_STEPS: %d" % self.LR_DECAY_STEPS)
        logger.info("----------------------------------------")