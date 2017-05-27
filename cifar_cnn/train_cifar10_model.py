import tensorflow as tf
import numpy as np
import time
import os
from datetime import datetime
import basic_cifar10_model as cifar10_model
import read_cifar10_data
from basic_param import *



def train():
    global_step = tf.Variable(0, trainable=False)

    train_images, train_labels = read_cifar10_data.cifar10_input(FLAGS.data_dir,
                                                                 FLAGS.batch_size,
                                                                 False,
                                                                 False)
    evalution_images, evalution_labels = read_cifar10_data.cifar10_input(FLAGS.data_dir,
                                                                 FLAGS.batch_size,
                                                                 True,
                                                                 False)
    # training
    y_predict = cifar10_model.inference(train_images)
    real_loss = cifar10_model.loss(train_labels, y_predict)
    train_step, lr = cifar10_model.solver(real_loss, global_step)
    acc = cifar10_model.accuarcy(train_labels, y_predict)

    # evalution
    y_predict_eval = cifar10_model.inference(evalution_images, evalution=True)
    acc_eval = cifar10_model.accuarcy(evalution_labels, y_predict_eval)

    tf.scalar_summary('loss', real_loss)
    tf.scalar_summary('accuracy', acc)
    summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver(tf.all_variables())

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir, sess.graph)
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        for step in xrange(NUM_STEP):
            start_time = time.time()
            [loss_value, acc_value, _, lr_value] = sess.run([real_loss, acc, train_step, lr])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step%10 == 0:
                format_str = ('%s: step %d, loss = %.5f, acc = %.5f, lr = %.5f (%.3f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, acc_value, lr_value,
                                    FLAGS.batch_size/duration, duration))

            if step%100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step%1000 == 0:
                [acc_eval_value] = sess.run([acc_eval])
                print("%s: evaluation acc = %.5f" % (datetime.now(), acc_eval_value))

            if step % 10000 == 0 or step + 1 == NUM_STEP:
                checkpoint_path = os.path.join(FLAGS.model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, step)


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
    tf.Variable