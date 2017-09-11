# encoding: utf-8

# Author: Shaoguang Cheng
# Email: shaoguang.csg@alibaba-inc.com

import tensorflow as tf
import numpy as np
from DLAPP.DLBase.activation import leaky_relu

# A
global_names = globals()
global_names['_layer_name_list'] = []
global_names['name_reuse'] = True


def flatten_reshape(variable, name=''):
    """
    convert a high dimension to a vector,
    :param variable: a tensorflow variable
    :param name: name for reshaped variable
    :return:
    """
    dim = 1
    for _ in variable.get_shape()[1:].as_list():
        dim *= _
    return tf.reshape(variable, shape=[-1, dim], name=name)


def get_variable_with_name(name, train_only=True, verbose=False):
    """
        return variable with the given name
    :param name:
    :param train_only: whether searching variables only in trainable variables
    :param printable:
    :return:
    """
    print(" Get variables {0}".format(name))
    vars = tf.trainable_variables() if train_only else tf.all_variables()
    var_list = [var for var in vars if name in var.name]

    if verbose:
        for index, var in enumerate(var_list):
            print("{:5} {:20}   {}".format(index, var.name, str(var.get_shape())))
    return var_list


def initialize_global_variables(sess=None):
    """Excute ``sess.run(tf.global_variables_initializer())`` for TF12+ or
    sess.run(tf.initialize_all_variables()) for TF11.
    Parameters
    ----------
    sess : a Session
    """
    assert sess is not None
    try:    # TF12
        sess.run(tf.global_variables_initializer())
    except: # TF11
        sess.run(tf.initialize_all_variables())


class Layer(object):
    """
    Layer is a mixin class.
    Base layer for all kinds if neural network layers.
    Because each layer can keep track of the layer(s) feeding into it, a network's output can double
    as a handle to the full network.
    """

    def __init__(self, inputs=None, name='Layer'):
        """
        :param inputs:
        :param name:
        """
        self.inputs = inputs
        if (name in global_names['_layer_name_list']) and (global_names['name_reuse'] == False):
            pass
        else:
            self.name = name
            if name not in ['', ' ', None, False]:
                global_names['_layer_name_list'].append(name)

    def __str__(self):
        return 'Layer {}'.format(self.__class__.__name__)

    def print_params(self, details=False):
        """
        print all parameters info in the network
        :param details:
        :return:
       """
        for index, param in enumerate(self.all_params):
            if details:
                try:
                    print("  param {:3}: {:15} (mean: {:<18}, median: {:<18}, std: {:<18})   {}".format(index, str(
                        param.eval().shape), param.eval().mean(), np.median(param.eval()), param.eval().std(), param.name))
                except:
                    raise Exception(
                        "Hint: print params details after sess.run(tf.initialize_all_variables()) or use network.print_params(False).")
            else:
                print("  param {:3}: {:15}    {}".format(index, str(param.get_shape()), param.name))

        print("  num of params: %d" % self.count_params())

    def print_layers(self):
        for index, layer in enumerate(self.all_layers):
            print("layer {0}, {1}".format(index, str(layer)))

    def count_params(self):
        """
        return the number of parameters in the network
        :return:
        """
        num = 0
        for param in self.all_params:
            num_tmp = 1
            for dim in param.get_shape():
                try :
                    dim = int(dim)
                except:
                    dim = 1
                if dim:
                    num_tmp *= dim
            num +=  num_tmp
        return num


class InputLayer(Layer):
    """
    InputLayer is the starting layer of a neural network.
    """
    def __init__(self, inputs=None, n_features=None, name ='input_layer'):
        super(InputLayer, self).__init__(inputs=inputs, name=name)
        print("  Instantiate InputLayer  %s: %s" % (self.name, inputs.get_shape()))
        self.outputs = inputs
        self.all_layers = []
        self.all_params = []
        self.all_dropout = {}


class DenseLayer(Layer):
    """
    FC layer
    """
    def __init__(self, layer=None, num_units=1000, act_func=tf.nn.relu,
                 w_init=tf.random_normal_initializer(stddev=0.01),
                 b_init=tf.constant_initializer(value=0.0),
                 w_init_args={},
                 b_init_args={},
                 name='dense_layer'):
        super(DenseLayer,self).__init__(name=name)
        self.inputs = layer.outputs

        assert self.inputs.get_shape().ndims == 2, 'The input dimension must be rank 2, please reshape or flatten it'

        input_dim = self.inputs.get_shape().as_list()[-1]
        self.num_units = num_units
        print("  Instantiate DenseLayer  %s: %d %s" % (self.name, num_units, act_func.__name__))

        with tf.variable_scope(name):
            W = tf.get_variable(name='W', shape=(input_dim, self.num_units), initializer=w_init, **w_init_args)
            if b_init:
                b = tf.get_variable(name='b', shape=(num_units), initializer=b_init, **b_init_args)
                self.outputs = act_func(tf.matmul(self.inputs, W)+b)
            else:
                self.outputs = act_func(tf.matmul(self.inputs, W))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_dropout = dict(layer.all_dropout)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])

class DropoutLayer(Layer):
    pass


class Conv2dLayer(Layer):
    """
    A 2D CNN layer

        Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function
        The function that is applied to the layer activations.
    shape : list of shape
        shape of the filters, [filter_height, filter_width, in_channels, out_channels].
    strides : a list of ints.
        The stride of the sliding window for each dimension of input.\n
        It Must be in the same order as the dimension specified with format.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable().
    b_init_args : dictionary
        The arguments for the biases tf.get_variable().
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(self, layer, act_func=tf.nn.relu, shape=[3,3,1,100], stride=[1,1,1,1], padding='SAME',
                 W_init=tf.truncated_normal_initializer(stddev=0.02),
                 b_init=tf.constant_initializer(value=0.0),
                 W_init_arg={},
                 b_init_arg={},
                 name='cnn_layer'):
        super(Conv2dLayer, self).__init__(name=name)
        self.inputs = layer.outputs

        print("  Instantiate Conv2dLayer %s: %s, %s, %s, %s" %
              (self.name, str(shape), str(stride), padding, act_func.__name__))

        with tf.variable_scope(name) as scope:
            W = tf.get_variable(name='W_con2d', shape=shape, initializer=W_init, **W_init_arg)
            if b_init:
                b = tf.get_variable(name='b_conv2d', shape=[shape[-1]], initializer=b_init, **b_init_arg)
                self.outputs = act_func(tf.nn.conv2d(self.inputs, W, strides=stride, padding=padding)+b)
            else:
                self.outputs = act_func(tf.nn.conv2d(self.inputs, W, strides=stride, padding=padding))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_dropout = dict(layer.all_dropout)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])

class PoolLayer(Layer):
    """
    The :class:`PoolLayer` class is a Pooling layer, you can choose
    ``tf.nn.max_pool`` and ``tf.nn.avg_pool`` for 2D or
    ``tf.nn.max_pool3d()`` and ``tf.nn.avg_pool3d()`` for 3D.
    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    ksize : a list of ints that has length >= 4.
        The size of the window for each dimension of the input tensor.
    strides : a list of ints that has length >= 4.
        The stride of the sliding window for each dimension of the input tensor.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    pool : a pooling function
        - class ``tf.nn.max_pool``
        - class ``tf.nn.avg_pool``
        - class ``tf.nn.max_pool3d``
        - class ``tf.nn.avg_pool3d``
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(self, layer, pool=tf.nn.max_pool, ksize=[1,2,2,1], stride=[1,2,2,1], padding='SAME', name='pool_layer'):
        super(PoolLayer, self).__init__(name=name)
        self.inputs = layer.outputs

        print("  Instantiate PoolLayer %s: %s, %s, %s, %s" %
              (self.name, str(ksize), str(stride), padding, pool.__name__))

        self.outputs = pool(self.inputs, ksize, stride, padding, name=name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_dropout = dict(layer.all_dropout)
        self.all_layers.extend([self.outputs])


class FlattenLayer(Layer):
    """
    The :class:`FlattenLayer` class is layer which reshape high-dimension
    input to a vector. Then we can apply DenseLayer, RNNLayer, ConcatLayer and
    etc on the top of it.
    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row * mask_col * n_mask]
    """
    def __init__(self, layer, name='faltten_layer'):
        super(FlattenLayer, self).__init__(name=name)
        self.inputs = layer.outputs

        self.outputs = flatten_reshape(self.inputs, name=name)

        print("  Instantiate FlattenLayer %s: %d" % (self.name, int(self.outputs.get_shape()[-1])))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_dropout = dict(layer.all_dropout)
        self.all_layers.extend([self.outputs])


class LeakyReluLayer(Layer):

    def __init__(self, layer, leakiness, name='leaky_relu_layer'):
        super(LeakyReluLayer, self).__init__(name=name)
        self.inputs = layer.outputs

        self.outputs = leaky_relu(self.inputs, leakiness)

        print("  Instantiate LeakyReluLayer %s: %d" % (self.name, int(self.outputs.get_shape()[-1])))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_dropout = dict(layer.all_dropout)
        self.all_layers.extend([self.outputs])


class BatchNormLayer(Layer):
    """
    Batch normalization on fully-connected or convolutional maps.
    Parameters
    -----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    decay : float, default is 0.9.
        A decay factor for ExponentialMovingAverage, use larger value for large dataset.
    epsilon : float
        A small float number to avoid dividing by 0.
    act : activation function.
    is_train : boolean
        Whether train or inference.
    beta_init : beta initializer
        The initializer for initializing beta
    gamma_init : gamma initializer
        The initializer for initializing gamma
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(self, layer, decay=0.9, epsilon=1e-6, act_func=tf.identity, is_train=False,
                 beta_init=tf.zeros_initializer,
                 gamma_init=tf.random_normal_initializer(mean=1.0,stddev=0.002),
                 name='batchnorm_layer'):
        super(BatchNormLayer, self).__init__(name=name)
        self.inputs = layer.outputs

        print("  Instantiate BatchNormLayer %s: decay: %f, epsilon: %f, act: %s, is_train: %s" %
                            (self.name, decay, epsilon, act_func.__name__, is_train))

        x_shape = self.inputs.get_shape()
        param_shape = x_shape[-1]

        from tensorflow.python.training import moving_averages

        with tf.variable_scope(name) as scope:
            print(name)
            beta = tf.get_variable('beta', shape=param_shape, initializer=beta_init, trainable=is_train)
            gamma = tf.get_variable('gama', shape=param_shape, initializer=gamma_init, trainable=is_train)

            global_mean = tf.get_variable('global_mean', shape=param_shape, initializer=tf.zeros_initializer,
                                          trainable=False)
            global_var = tf.get_variable('global_var', shape=param_shape, initializer=tf.constant_initializer(1.0))

            axis = range(len(x_shape)-1)
            mean,var = tf.nn.moments(self.inputs, axis)

            update_global_mean = moving_averages.assign_moving_average(global_mean, mean, decay)
            update_global_var = moving_averages.assign_moving_average(global_var, var, decay)

            def mean_var_update():
                with tf.control_dependencies([update_global_mean, update_global_var]):
                    return tf.identity(mean), tf.identity(var)

            if is_train:
                mean, var = mean_var_update()
                self.outputs = act_func(tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon))
            else:
                self.outputs = act_func(tf.nn.batch_normalization(self.inputs, global_mean, global_var, beta, gamma, epsilon))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_dropout = dict(layer.all_dropout)
        self.all_layers.extend([self.outputs])
        self.all_params.extend([beta, gamma, global_mean, global_var])


if __name__ == '__main__':
    x = tf.Variable([1,2,3])
    l = InputLayer(x, name='l1')

    print(l.print_layers())


