# encoding: utf-8

# Author: Shaoguang Cheng
# Email: shaoguang.csg@alibaba-inc.com

import yaml

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_conf_file', 'conf/resnet_cifar_conf.yaml', 'Path to the model config yaml file')

# singleton pattern
def singleton(cls):
    _instances = {}

    def _singleton(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return _singleton

@singleton
class ModelConf(object):

    def __init__(self):
        with open(FLAGS.model_conf_file) as model_conf_file:
            model_conf = yaml.load(model_conf_file)

        self.num_step = model_conf.get('num_step', 100000)
        self.leakiness = model_conf.get('leakiness', 0.01)
        self.use_bottleneck = model_conf.get('use_bottleneck', True)
        self.num_residual_units = model_conf.get('num_residual_units', 5)
        self.num_classes = model_conf.get('num_classes', 10)
        self.batch_size = model_conf.get('batch_size', 100)
        self.weight_decay_rate = model_conf.get('weight_decay_rate', 0.01)
        self.lrn_rate = model_conf.get('lrn_rate', 0.002)
        self.lr_decay_steps = model_conf.get('lr_decay_steps', 20000)
        self.optimizer = model_conf.get('optimizer', 'sgd')

        self.dataset = model_conf.get('dataset', 'cifar10')
        self.data_path = model_conf.get('data_path', '/Users/cheng/Data/cifar10_data/cifar-10-batches-bin/')
        self.train_dir = model_conf.get('train_dir', None)
        self.mode = model_conf.get('mode', 'train')
        self.eval_batch_count = model_conf.get('eval_batch_count', 50)
        self.model_path = model_conf.get('model_path', None)


        self._check_param()

    def _check_param(self):
        pass
