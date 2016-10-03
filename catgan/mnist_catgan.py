import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchmat_to_tileimg
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/vae',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 20,
        """number of epochs to run""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

# leaky relu
def lrelu(x, leak=0.1, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5*(1 + leak)
        f2 = 0.5*(1 - leak)
        return f1*x + f2*abs(x)

# discriminator
x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1,28,28,1])
D = conv(x_img, 32, [5, 5], [1, 1], activation_fn=lrelu)
D = pool(D, [3, 3], [2, 2])
D = conv(D, 64, [3, 3], [1, 1], activation_fn=lrelu)
D = conv(D, 64, [3, 3], [1, 1], activation_fn=lrelu)
D = pool(D, [3, 3], [2, 2])
D = conv(D, 128, [3, 3], [1, 1], activation_fn=lrelu)
D = conv(D, 10, [1, 1], [1, 1], activation_fn=lrelu)
D = fc(flat(D), 128, activation_fn=lrelu)
D = fc(D, 10, activation_fn=tf.nn.softmax)

# generator
z = tf.placeholder(tf.float32, [None, 128])
G = fc(z, 8*8*96, activation_fn=lrelu)
G = tf.reshape(G, [-1, 8, 8, 96])

saver = tf.train.Saver()
sess = tf.Session()

def train():
    raise NotImplementedError

def test():
    raise NotImplementedError

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
