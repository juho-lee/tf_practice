import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/mnist", one_hot=True)
num_train = mnist.train.num_examples
num_test = mnist.test.num_examples
batch_size = 100
num_train_batches = num_train / batch_size
num_test_batches = num_test / batch_size
num_epochs = 10

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_img = tf.reshape(x, [-1,28,28,1])
conv1 = layers.max_pool2d(
        layers.convolution2d(x_img, 32, [5, 5]),
        [2, 2], [2, 2])
conv2 = layers.max_pool2d(
        layers.convolution2d(conv1, 64, [5, 5]),
        [2, 2], [2, 2])
y = layers.fully_connected(layers.flatten(conv2),
        10, activation_fn=tf.nn.softmax)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

import time
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_epochs):
        start = time.time()
        train_acc = 0.
        for j in range(num_train_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            feed_dict = {x:batch_x, y_:batch_y}
            sess.run(train_step, feed_dict=feed_dict)
            train_acc += sess.run(accuracy, feed_dict=feed_dict)
        train_acc /= num_train_batches
        test_acc = 0.
        for j in range(num_test_batches):
            batch_x, batch_y = mnist.test.next_batch(batch_size)
            feed_dict = {x:batch_x, y_:batch_y}
            test_acc += sess.run(accuracy, feed_dict=feed_dict)
        test_acc /= num_test_batches
        print "Epoch %d (%f sec), train acc %f, test acc %f" \
                % ( i+1, time.time()-start, train_acc, test_acc)
