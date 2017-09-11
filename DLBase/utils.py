# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from DLAPP.models.slim.nets.vgg import vgg_16
import numpy as np

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_folder, output_node_names):
    """
    freeze saved graph
    :param model_folder:
    :param output_node_names
    :return:
    """

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes
    # freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点
    # 输出结点可以看我们模型的定义
    # 只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃
    # 所以,output_node_names必须根据不同的网络进行修改
#    output_node_names = "Accuracy/predictions"

    clear_devices = True
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    # 这边已经将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen
    # 相当于将参数已经固化在了图当中
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constant
        # If you have a trained graph containing Variable ops,
        # it can be convenient to convert them all to Const ops holding the same values.
        # This makes it possible to describe the network fully with a single GraphDef file,
        # and allows the removal of a lot of ops related to loading and saving the variables.
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def load_graph(frozen_graph_filename):
    """
    load graph from frozen_graph_filename
    :param frozen_graph_filename:
    :return:
    """
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def load_pretrained_model(model_name, saved_checkpoint):
    batch_size = 1
    height, width = 224,224
    num_classes = 1000
    inputs = tf.placeholder(tf.float32, shape=(None, height, width, 3))

    logits, end_points = vgg_16(inputs, num_classes)
#    graph = tf.get_default_graph()
    saver = tf.train.Saver()
    with tf.Session() as session:
#        session.run(tf.global_variables_initializer())
#        W_fc7_1 = session.run(graph.get_tensor_by_name('vgg_16/fc7/weights:0'))
#        print(W_fc7_1)

        saver.restore(session, model_name) # load pretrained
        saver.save(session, saved_checkpoint+'vgg_16')
        print("Model Saved: %s", model_name)

if __name__ == "__main__":
    model_name = "/Users/cheng/Data/pretrained_models/vgg_16/vgg_16.ckpt"
    saved_checkpoint = '/Users/cheng/Data/pretrained_models/vgg_16/vgg_checkpoint/'
    output_node_names = 'vgg_16/fc8/squeezed'

#    load_pretrained_model(model_name, saved_checkpoint)
#    freeze_graph(saved_checkpoint, output_node_names)

    graph = load_graph(saved_checkpoint+'frozen_model.pb')
    for op in graph.get_operations():
        print(op.name, op.values())

    x = graph.get_tensor_by_name("prefix/Placeholder:0")
    y = graph.get_tensor_by_name("prefix/vgg_16/fc8/squeezed:0")

    session = tf.Session(graph=graph)
    res = session.run(y, feed_dict={x: np.random.random((1,224,224,3)) })
    print(len(res[0]))