import tensorflow as tf
import numpy as np

def repeat(x, n):
    m = tf.shape(x)[0]
    return tf.reshape(tf.tile(tf.reshape(x, [-1, 1]), [1, n]), tf.pack([m, -1]))

x_ = tf.placeholder(tf.float32, [None, 3])

x = np.random.rand(2, 3)
print x

sess = tf.Session()
z = sess.run(repeat(x_, 2), {x_:x})
print z


