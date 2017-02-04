import tensorflow as tf

#original image size from cifar-10 32x32
IMAGE_SIZE = 32

NUM_TRAINING_IMAGE = 50000
NUM_EVALUATING_IMAGE = 10000
NUM_CLASS = 10
LR_DECAY_STEP = 1200
NUM_STEP = 100000

# some basic parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../data/cifar10_data/cifar-10-batches-bin/',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_string('model_dir', '../model/cifar10_model/',
                           """Path to the save CIFAR-10 Model checkpoint.""")
tf.app.flags.DEFINE_string('summary_dir', '../summary/cifar10_summary/',
                           """Path to the save CIFAR-10 summery.""")