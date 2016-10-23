import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils.nn import *
import time

mnist = input_data.read_data_sets('data/mnist', one_hot=True)
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_test_batches = mnist.test.num_examples / batch_size

x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

is_tr = tf.placeholder(tf.bool)
with tf.variable_scope('cls'):
    net = pool(conv_bn(x_img, 16, [3, 3], is_tr), [2, 2])
    net = pool(conv_bn(net, 32, [3, 3], is_tr), [2, 2])
    net = pool(conv_bn(net, 64, [3, 3], is_tr), [2, 2])
    net = fc_bn(flat(net), 10, is_tr, activation_fn=None)
cent, acc = get_classification_loss(net, y)

vars = tf.all_variables()
vars = [var for var in vars if 'cls' in var.name]
train_op = get_train_op(cent, var_list=vars)
saver = tf.train.Saver(vars)

from utils.misc import Logger
train_Logger = Logger('train cent', 'train acc')
test_Logger = Logger('test cent', 'test acc')

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(3):
    start = time.time()
    train_Logger.clear()
    for j in range(n_train_batches):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        train_Logger.accum(sess.run([train_op, cent, acc],
            {x:batch_x, y:batch_y, is_tr:True}))
    test_Logger.clear()
    for j in range(n_test_batches):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        test_Logger.accum(sess.run([cent, acc],
            {x:batch_x, y:batch_y, is_tr:False}))

    print (train_Logger.get_status(i+1, time.time()-start) + \
            test_Logger.get_status_no_header())
saver.save(sess, 'temp.ckpt')

with tf.variable_scope('cls', reuse=True):
    net2 = pool(conv_bn(x_img, 16, [3, 3], is_tr), [2, 2])
    net2 = pool(conv_bn(net2, 32, [3, 3], is_tr), [2, 2])
    net2 = pool(conv_bn(net2, 64, [3, 3], is_tr), [2, 2])
    net2 = fc_bn(flat(net2), 10, is_tr, activation_fn=None)
cent2, acc2 = get_classification_loss(net2, y)

test_Logger.clear()
for j in range(n_test_batches):
    batch_x, batch_y = mnist.test.next_batch(batch_size)
    test_Logger.accum(sess.run([cent2, acc2],
        {x:batch_x, y:batch_y, is_tr:False}))
print test_Logger.get_status_no_header()

