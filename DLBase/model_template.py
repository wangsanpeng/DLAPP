import tensorflow as tf
from functools import wraps
import numpy as np


def lazy_property(func):
    attribute = '_cache_' + func.__name__

    @property
    @wraps(func)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return decorator


class Model(object):
    """
    A demo to show how to organize the model with tensorflow
    """
    def __init__(self, data, target, mode):
        self.data = data
        self.target = target
        self.mode = mode
        self.build_model
        self.optimize
        self.error
        self.loss
        self.W

#    @lazy_property
#    def build_model(self):
#        if self.mode == 'train':
#            print "train"
#            reuse = False
#        else:
#            print 'test'
#            reuse = True

#        with tf.variable_scope("xxxx", reuse=reuse):
#            W = tf.get_variable('w', initializer=tf.truncated_normal([1]), dtype=tf.float32)
#            b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[1]), dtype=tf.float32)
#        self._W = [W, b]
#        return W * self.data + b

    @lazy_property
    def build_model(self):
        W = tf.Variable(np.random.uniform(-0.1, 0.1, [1]), dtype=tf.float32)
        b = tf.Variable(np.zeros([1]), dtype=tf.float32)
        self._W = [W, b]
        return W * self.data + b


    @lazy_property
    def optimize(self):
        self._loss = tf.reduce_mean(tf.square(self.target - self.build_model))
        optimizer = tf.train.AdamOptimizer(0.1)
        return optimizer.minimize(self._loss)

    @lazy_property
    def W(self):
        return self._W

    @lazy_property
    def loss(self):
        return self._loss

    @lazy_property
    def error(self):
        pass

    tf.nn.sigmoid_cross_entropy_with_logits()

if __name__ == '__main__':
    # real data
    x_data = np.random.rand(10).astype(np.float32).reshape([10,1])
    y_data = 0.3 * x_data + 0.1

    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, 1])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    # build model
    train_model = Model(x, y, "train")

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for step in range(201):
        sess.run(train_model.optimize, feed_dict={x:x_data, y:y_data})
#        print sess.run(train_model.loss, feed_dict={x:x_data, y:y_data})
        print sess.run(train_model.build_model, feed_dict={x:x_data, y:y_data})


