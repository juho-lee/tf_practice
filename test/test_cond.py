import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 4])
y = tf.select(tf.equal(x,0), 
        tf.fill(tf.shape(x), 4.), 
        tf.square(x))

sess = tf.Session()
x_ = np.random.normal(size=(5, 4))
x_ = x_ + abs(x_)

print x_
print sess.run(y, {x:x_})
