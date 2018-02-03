import tensorflow as tf
from rbm import RBM

tf.flags.DEFINE_integer("epochs", 100, "")
tf.flags.DEFINE_integer("batch_size", 50, "")
tf.flags.DEFINE_integer("num_hidden", 100, "")
tf.flags.DEFINE_float("momentum", 0.9, "")
tf.flags.DEFINE_float("v_lr", 0.1, "")
tf.flags.DEFINE_float("w_lr", 0.1, "")
tf.flags.DEFINE_float("h_lr", 0.1, "")
tf.flags.DEFINE_string("data_path", "ml-100k/u1.base", "")
tf.flags.DEFINE_string("sep", "\t", "")
FLAGS = tf.flags.FLAGS


if __name__ == "__main__":

    rbm = RBM(FLAGS.num_hidden)
    rbm.fit(data_path=FLAGS.data_path, sep=FLAGS.sep, batch_size=FLAGS.batch_size,
            w_lr=FLAGS.w_lr, v_lr=FLAGS.v_lr, h_lr=FLAGS.h_lr, momentum=FLAGS.momentum)
