import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

x_ = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 4])

x = np.random.rand(2,1)
y = np.random.rand(2,4)

sess = tf.Session()
print x
print y
print np.tile(x,[1,4]) + y
print sess.run(x_ + y_, {x_:x, y_:y})

