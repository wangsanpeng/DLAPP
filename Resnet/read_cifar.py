"""
For the introduction of cifar-10 and cifar-100, http://www.cs.toronto.edu/~kriz/cifar.html
In this file, I will train a cifar model from scratch
"""

import tensorflow as tf
import os

IMAGE_SIZE = 32
NUM_TRAINING_IMAGE = 50000
NUM_EVALUATING_IMAGE = 10000

def cifar_input(dataset, data_path, batch_size, evaluation, do_augmentation=False):
    """
    :param data_dir:  data path
    :param batch_size:
    :param evaluation: whether training or evaluation process
    :return: batch of cifar-10 data (image, label)
    """
    if dataset == 'cifar10':
        label_bytes = 1
        label_offset = 0
        num_classes = 10
    elif dataset == 'cifar100':
        label_bytes = 1
        label_offset = 1
        num_classes = 100
    else:
        raise ValueError('Not supported dataset %s', dataset)

    if not evaluation:
        filenames = [os.path.join(data_path, "data_batch_%d.bin") % i for i in xrange(1, 6)]
        num_image_per_epoch = NUM_TRAINING_IMAGE
    else:
        filenames = [os.path.join(data_path, "test_batch.bin")]
        num_image_per_epoch = NUM_EVALUATING_IMAGE

    for f in filenames:
        if not os.path.exists(f):
            raise ValueError("Failed to find the file" + f)

    filename_queue = tf.train.string_input_producer(filenames)
    cifar_image = read_cifar10(filename_queue, label_bytes, label_offset)
    image = tf.cast(cifar_image.uint8_image, tf.float32)

    if do_augmentation:
        image = tf.image.resize_image_with_crop_or_pad(
            image, IMAGE_SIZE + 4, IMAGE_SIZE + 4)
        image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_brightness(image, max_delta=63.0/255.0)
        image = tf.image.random_contrast(image,lower=0.2, upper=1.8)
#        image = tf.image.per_image_whitening(image) # for 0.11
        image = tf.image.per_image_standardization(image)
    else:
#        image = tf.image.per_image_whitening(image)# for 0.11
        image = tf.image.per_image_standardization(image)

    min_faction_examples_in_queue = 0.4
    min_queue_examples = int(min_faction_examples_in_queue*num_image_per_epoch)

    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    if evaluation:
        return _generate_example_label_batch(image, cifar_image.label, min_queue_examples, batch_size, num_classes, shuffle=False)
    else:
        return _generate_example_label_batch(image, cifar_image.label, min_queue_examples, batch_size, num_classes, shuffle=True)


def read_cifar10(filename_queue, label_bytes, label_offset):
    """
    :param filename_queue:
    :return: reutrn a image and label from cifar filename queue
    """
    class Cifar10Record(object):
        pass
    result = Cifar10Record()

    result.height = IMAGE_SIZE
    result.width = IMAGE_SIZE
    result.depth = 3
    image_bytes = result.height*result.width*result.depth
    record_bytes = image_bytes+label_bytes+label_offset

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(tf.slice(record_bytes, [label_offset], [label_bytes]), tf.int32)
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [result.depth, result.height, result.width])
    result.uint8_image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_example_label_batch(image, label, min_queue_examples, batch_size, num_classes, shuffle=True):
    """
    :param images:
    :param label:
    :param min_queue_examples:
    :param batch_size:
    :param shuffle:
    :return:
    """
    num_preprocess_threads = 16

    if shuffle:
        batch_image, batch_label = tf.train.shuffle_batch([image, label],
                                                          batch_size=batch_size,
                                                          num_threads=num_preprocess_threads,
                                                          capacity=min_queue_examples+batch_size*3,
                                                          min_after_dequeue=min_queue_examples,
                                                          enqueue_many=False)
    else:
        batch_image, batch_label = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=num_preprocess_threads,
                                                  capacity=min_queue_examples+batch_size*3)

    print batch_label.get_shape()

    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    batch_label = tf.sparse_to_dense(
        tf.concat(1, [indices, batch_label]),
        [batch_size, num_classes], 1.0, 0.0)

    assert len(batch_image.get_shape()) == 4
    assert batch_image.get_shape()[0] == batch_size
    assert batch_image.get_shape()[-1] == 3
    assert len(batch_label.get_shape()) == 2
    assert batch_label.get_shape()[0] == batch_size
    assert batch_label.get_shape()[1] == num_classes

    tf.image_summary(batch_image.op.name+'_images', batch_image)

    return batch_image, batch_label

from DLAPP.DLBase.visualize import *
if __name__ == '__main__':
    filename = '/Users/cheng/Data/cifar10_data/cifar-10-batches-bin/'
    batch = cifar_input("cifar100", filename, 10, True, False)
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        b = sess.run(batch[0])
        print batch[0].get_shape()
        images2d(b,saveable=False, second=30)
