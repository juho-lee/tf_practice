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

is_tr = tf.placeholder(tf.bool)
net = pool(conv_bn(x_img, 16, [3, 3], is_tr), [2, 2])
net = pool(conv_bn(net, 32, [3, 3], is_tr), [2, 2])
feat = pool(conv_bn(net, 64, [3, 3], is_tr), [2, 2])
net = fc_bn(flat(feat), 10, is_tr, activation_fn=None)
cent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, y))
train_op = get_train_op(cent)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(net, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())
n_epochs = 3

import time
for i in range(n_epochs):
    train_cent = 0.
    train_acc = 0.
    start = time.time()
    for j in range(n_train_batches):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, batch_cent, batch_acc = sess.run([train_op, cent, acc],
                {x:batch_x, y:batch_y, is_tr:True})
        train_cent += batch_cent
        train_acc += batch_acc
    train_cent /= n_train_batches
    train_acc /= n_train_batches

    valid_cent = 0.
    valid_acc = 0.
    for j in range(n_valid_batches):
        batch_x, batch_y = mnist.validation.next_batch(batch_size)
        batch_cent, batch_acc = sess.run([cent, acc],
                {x:batch_x, y:batch_y, is_tr:False})
        valid_cent += batch_cent
        valid_acc += batch_acc
    valid_cent /= n_valid_batches
    valid_acc /= n_valid_batches

    line = 'epoch %d (%f), train cent %f, train acc %f, valid cent %f, valid acc %f' \
            % (i, time.time()-start, train_cent, train_acc, valid_cent, valid_acc)
    print line

batch_x, _ = mnist.test.next_batch(100)
from utils.image import batchmat_to_tileimg, batchimg_to_tileimg
import matplotlib.pyplot as plt
plt.figure('original')
plt.gray()
plt.axis('off')
plt.imshow(batchmat_to_tileimg(batch_x, (28, 28), (10, 10)))

batch_feat = sess.run(feat, {x:batch_x, is_tr:False})
plt.figure('first feature')
plt.gray()
plt.axis('off')
plt.imshow(batchimg_to_tileimg(batch_feat[:,:,:,0:1], (10, 10)))
plt.show()
        
        

