import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from DLAPP.Resnet import parse_args, read_cifar, resnet_cifar

tf.logging.set_verbosity(tf.logging.INFO)

def train(model_conf):
    """train resnet"""
    images, labels = read_cifar.cifar_input(model_conf.dataset,
                                            model_conf.data_path,
                                            model_conf.batch_size,
                                            evaluation=False,
                                            do_augmentation=True)
    model = resnet_cifar.ResNet(model_conf, images, labels)
    model.build_graph()

    y = tf.argmax(model.labels, 1)
    y_ = tf.argmax(model.predictions, 1)
    acc = tf.reduce_mean(tf.to_float(tf.equal(y, y_)))

    tf.scalar_summary('accuracy', acc)
    summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver(tf.all_variables())

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(model_conf.train_dir, sess.graph)
        tf.summary.FileWriter
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        for step in xrange(model_conf.num_step):
            start_time = time.time()
            [loss_value, acc_value, _, lr_value] = sess.run([model.loss, acc, model.train_op, model.lrn_rate])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step%10 == 0:
                format_str = ('%s: step %d, loss = %.5f, acc = %.5f, lr = %.5f (%.3f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, acc_value, lr_value,
                                    model_conf.batch_size/duration, duration))

            if step%100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 10000 == 0 or step + 1 == model_conf.num_step:
                checkpoint_path = os.path.join(model_conf.model_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, step)

def evaluate(model_conf):
    images, labels = read_cifar.cifar_input(model_conf.dataset,
                                            model_conf.data_path,
                                            model_conf.batch_size,
                                            evaluation=True,
                                            do_augmentation=False)
    model = resnet_cifar.ResNet(model_conf, images, labels)
    model.build_graph()

    saver = tf.train.Saver()
    sess = tf.Session()
    tf.train.start_queue_runners(sess)

    try:
        ckpt_state = tf.train.get_checkpoint_state(model_conf.model_path)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Can not restore checkpoint %s', e)
        return

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', model_conf.model_path)
        return

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    total_prediction, correct_prediction = 0, 0
    best_precision = 0.0
    for _ in xrange(model_conf.eval_batch_count):
        (loss, y, y_) = sess.run([model.loss, model.labels, model.predictions])

        y = np.argmax(y, 1)
        y_ = np.argmax(y_, 1)
        correct_prediction += np.sum(y == y_)
        total_prediction += y_.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f\n' %
                    (loss, precision, best_precision))


def main(_):
    model_conf = parse_args.ModelConf()
    if model_conf.mode == 'train':
        train(model_conf)
    elif model_conf.mode == 'eval':
        evaluate(model_conf)

if __name__ == '__main__':
    tf.app.run()
