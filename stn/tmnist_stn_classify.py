import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from utils.data import load_pkl
from spatial_transformer import spatial_transformer
import time

conv = layers.convolution2d
pool = layers.max_pool2d
fc = layers.fully_connected

x = tf.placeholder(tf.float32, [None, 60*60])
x_tensor = tf.reshape(x, [-1, 60, 60, 1])
y = tf.placeholder(tf.int32, [None])
y_one_hot = layers.one_hot_encoding(y, 10)

# localization net
loc = pool(x_tensor, [2, 2])
loc = conv(loc, 20, [5, 5], padding='VALID')
loc = pool(loc, [2, 2])
loc = conv(loc, 20, [5, 5], padding='VALID')
loc = fc(layers.flatten(loc), 50)
loc = fc(loc, 6, activation_fn=None,
        weights_initializer=tf.constant_initializer(np.zeros((50, 6))),
        biases_initializer=tf.constant_initializer(
            np.array([[1.,0,0],[0,1.,0]]).flatten()))

# classification net
trans = spatial_transformer(x_tensor, loc, 3.0)
cl = conv(trans, 32, [3, 3], padding='VALID')
cl = pool(cl, [2, 2])
cl = conv(trans, 32, [3, 3], padding='VALID')
cl = pool(cl, [2, 2])
cl = fc(layers.flatten(cl), 256)
y_logits = fc(cl, 10, activation_fn=None)

cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y_logits, y_one_hot))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))

train_xy, _, test_xy = load_pkl('data/tmnist/tmnist.pkl.gz')
train_x, train_y = train_xy
test_x, test_y = test_xy

batch_size = 100
n_train_batches = len(train_x)/batch_size
n_test_batches = len(test_x)/batch_size

n_epochs = 10
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(n_epochs):
        start = time.time()
        train_acc = 0.
        for j in range(n_train_batches):
            batch_x = train_x[j*batch_size:(j+1)*batch_size]
            batch_y = train_y[j*batch_size:(j+1)*batch_size]
            _, batch_acc = sess.run([train_step, accuracy],
                    {x:batch_x, y:batch_y})
            train_acc += batch_acc
        train_acc /= n_train_batches

        test_acc = 0.
        for j in range(n_test_batches):
            batch_x = test_x[j*batch_size:(j+1)*batch_size]
            batch_y = test_y[j*batch_size:(j+1)*batch_size]
            batch_acc = sess.run(accuracy, {x:batch_x, y:batch_y})
            test_acc += batch_acc
        test_acc /= n_test_batches

        print "Epoch %d (%f sec), train acc %f, test acc %f" \
                % (i+1, time.time()-start, train_acc, test_acc)
