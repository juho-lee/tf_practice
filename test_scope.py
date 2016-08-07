import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

def linear(x, dim, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        return layers.fully_connected(x, dim, activation_fn=None,
                weights_initializer=tf.random_normal_initializer())

x_ = tf.placeholder(tf.float32, [None, 10])
y1_ = linear(x_, 3, "test")
y2_ = linear(x_, 3, "test", reuse=True)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    x = np.random.rand(1, 10)
    y1, y2 = sess.run([y1_, y2_], feed_dict={x_:x})
    print y1
    print
    print y2
