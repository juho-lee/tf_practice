import tensorflow as tf
import numpy as np
from utils.nn import *
from utils.data import load_pkl
import time

train_xy, test_xy, _ = load_pkl('data/dmnist/dmnist_easy.pkl.gz')
batch_size = 100
train_x, train_y = train_xy
test_x, test_y = test_xy
n_train_batches = len(train_x)/batch_size
n_test_batches = len(test_x)/batch_size

x = tf.placeholder(tf.float32, [None, 32*32])
x_img = tf.reshape(x, [-1, 32, 32, 1])
y = tf.placeholder(tf.int32, [None])
y_one_hot = one_hot(y, 15)
is_training = tf.placeholder(tf.bool)

net = pool(conv(x_img, 16, [5, 5]), [2, 2])
net = pool(conv_bn(net, 32, [5, 5], is_training), [2, 2])
net = pool(conv_bn(net, 64, [5, 5], is_training), [2, 2])
net = pool(conv_bn(net, 128, [5, 5], is_training), [2, 2])
net = linear(flat(net), 15)
cent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, y_one_hot))
train_op = get_train_op(cent)
correct = tf.equal(tf.argmax(y_one_hot, 1), tf.argmax(net, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

n_epochs = 10
idx = range(len(train_x))
for i in range(n_epochs):
    train_cent = 0.
    train_acc = 0.
    start = time.time()
    np.random.shuffle(idx)
    for j in range(n_train_batches):
        batch_x = train_x[idx[j*batch_size:(j+1)*batch_size]]
        batch_y = train_y[idx[j*batch_size:(j+1)*batch_size]]
        feed_dict = {x:batch_x, y:batch_y, is_training:True}
        _, batch_cent, batch_acc = sess.run([train_op, cent, acc], feed_dict)
        train_cent += batch_cent
        train_acc += batch_acc
    train_cent /= n_train_batches
    train_acc /= n_train_batches

    test_acc = 0
    for j in range(n_test_batches):
        batch_x = test_x[j*batch_size:(j+1)*batch_size]
        batch_y = test_y[j*batch_size:(j+1)*batch_size]
        feed_dict = {x:batch_x, y:batch_y, is_training:False}
        test_acc += sess.run(acc, feed_dict)
    test_acc /= n_test_batches

    line = 'Epoch %d (%.3f secs), train cent %f, train acc %f, test acc %f' \
            %(i+1, time.time()-start, train_cent, train_acc, test_acc)
    print line


