import numpy as np
import tensorflow as tf

def msra_initializer(n_inputs):
    """
    :param n_inputs:
    :return:
    """
    stddev = np.sqrt(2.0/n_inputs)
    return tf.truncated_normal_initializer(stddev=stddev)