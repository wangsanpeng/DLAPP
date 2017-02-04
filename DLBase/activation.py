
import tensorflow as tf

def leaky_relu(x, leakiness=0.0):
    """
    return x if x > 0 else leakiness*x
    :param x:
    :param leakiness:
    :return:
    """
    return tf.select(tf.greater(x, 0.0), x, leakiness*x, name='leaky_relu')
