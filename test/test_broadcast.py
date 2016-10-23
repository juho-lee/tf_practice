import tensorflow as tf
import numpy as np

c = tf.placeholder(tf.float32, [None, 3])
x = tf.placeholder(tf.float32, [None, 2, 2, 3])

y = tf.reshape(c, [-1, 1, 1, 3]) * x

sess = tf.Session()

c_ = np.random.rand(3, 3)
x_ = np.random.rand(3, 2, 2, 3)

y_ = sess.run(y, {x:x_, c:c_})

