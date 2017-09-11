import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np

# test variable resuse

def _conv2d(x, output_channel, kernel_size, name='conv2d'):
    with tf.variable_scope(name) as scope:
        return layers.conv2d(x, output_channel, kernel_size=kernel_size, stride=1, padding='SAME',
                        activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.1),
                        reuse=True, scope=scope)

if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3, 2])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3, 2])
    cx = _conv2d(x, 3, 3)
    cy = _conv2d(y, 3, 3)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    xx = np.random.rand(2, 3, 3, 2)
    cx_1 = sess.run([cx], feed_dict={x:xx})

    cy_1 = sess.run([cy], feed_dict={y:xx})

    print(cx_1)
    print(cy_1)