import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils.nn import *
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('data/mnist', one_hot=True)
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_valid_batches = mnist.validation.num_examples / batch_size

x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# without BN
net = pool(conv(x_img, 16, [3, 3]), [2, 2])
net = pool(conv(net, 32, [3, 3]), [2, 2])
net = pool(conv(net, 64, [3, 3]), [2, 2])
net = linear(flat(net), 10)
cent = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(net, y))
train_op = get_train_op(cent)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(net, 1))
acc = tf.reduce_sum(tf.cast(correct, tf.float32))

# with BN
is_tr = tf.placeholder(tf.bool)
net_BN = pool(conv_bn(x_img, 16, [3, 3], is_tr), [2, 2])
net_BN = pool(conv_bn(net_BN, 32, [3, 3], is_tr), [2, 2])
net_BN = pool(conv_bn(net_BN, 64, [3, 3], is_tr), [2, 2])
net_BN = fc_bn(flat(net_BN), 10, is_tr, activation_fn=None)
cent_BN = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(net_BN, y))
train_op_BN = get_train_op(cent_BN)
correct_BN = tf.equal(tf.argmax(y, 1), tf.argmax(net_BN, 1))
acc_BN = tf.reduce_sum(tf.cast(correct_BN, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())
n_epochs = 10
print 'without BN:'
for i in range(n_epochs):
    train_cent = 0.
    train_acc = 0.
    for j in range(n_train_batches):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, batch_cent, batch_acc = sess.run([train_op, cent, acc], {x:batch_x, y:batch_y})
        train_cent += batch_cent
        train_acc += batch_acc
    train_cent /= n_train_batches
    train_acc /= n_train_batches

    valid_cent = 0.
    valid_acc = 0.
    for j in range(n_valid_batches):
        batch_x, batch_y = mnist.validation.next_batch(batch_size)
        batch_cent, batch_acc = sess.run([cent, acc], {x:batch_x, y:batch_y})
        valid_cent += batch_cent
        valid_acc += batch_acc
    valid_cent /= n_valid_batches
    valid_acc /= n_valid_batches

    line = 'epoch %d, train cent %f, train acc %f, valid cent %f, valid acc %f' \
            % (i, train_cent, train_acc, valid_cent, valid_acc)
    print line

print '\nwith BN:'
for i in range(n_epochs):
    train_cent = 0.
    train_acc = 0.
    for j in range(n_train_batches):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, batch_cent, batch_acc = sess.run([train_op_BN, cent_BN, acc_BN],
                {x:batch_x, y:batch_y, is_tr:True})
        train_cent += batch_cent
        train_acc += batch_acc
    train_cent /= n_train_batches
    train_acc /= n_train_batches

    valid_cent = 0.
    valid_acc = 0.
    for j in range(n_valid_batches):
        batch_x, batch_y = mnist.validation.next_batch(batch_size)
        batch_cent, batch_acc = sess.run([cent_BN, acc_BN],
                {x:batch_x, y:batch_y, is_tr:False})
        valid_cent += batch_cent
        valid_acc += batch_acc
    valid_cent /= n_valid_batches
    valid_acc /= n_valid_batches

    line = 'epoch %d, train cent %f, train acc %f, valid cent %f, valid acc %f' \
            % (i, train_cent, train_acc, valid_cent, valid_acc)
    print line
