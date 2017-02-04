import numpy as np
import tensorflow as tf

def msra_initializer(n_inputs):
    """
    :param n_inputs:
    :return:
    """
    stddev = np.sqrt(2.0/n_inputs)
    return tf.truncated_normal_initializer(stddev=stddev)


def _xaiver_initializer(n_inputs, n_outputs, uniform=True):
    """Set the parameter initialization using the method described.
      This method is designed to keep the scale of the gradients roughly the same
      in all layers.
      Xavier Glorot and Yoshua Bengio (2010):
               Understanding the difficulty of training deep feedforward neural
               networks. International conference on artificial intelligence and
               statistics.
      Args:
        n_inputs: The number of input nodes into each output.
        n_outputs: The number of output nodes for each input.
        uniform: If true use a uniform distribution, otherwise use a normal.
      Returns:
        An initializer.
      """
    if uniform:
        init = np.sqrt(6.0 / (n_inputs+n_outputs))
        return tf.random_uniform_initializer(-init, init)
    else:
        stddev = np.sqrt(3.0/(n_inputs+n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)