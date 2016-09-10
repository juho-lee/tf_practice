import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from spatial_transformer import *

"""
h = 4
w = 3

x = tf.expand_dims(repeat(tf.range(h), w), 1)
y = tf.expand_dims(tf.tile(tf.range(w), [h]), 1)
ind = tf.concat(1, [x, y])

ht = 6
wt = 5
tx = tf.clip_by_value(3, 0, ht-h)
ty = tf.clip_by_value(4, 0, wt-w)
tind = tf.concat(1, [x+tx, y+ty])
val = tf.ones(h*w)
T = tf.sparse_to_dense(tind, [ht, wt], val)
"""

h = 4
w = 3
n = 2
x = tf.tile(tf.expand_dims(repeat(tf.range(h), w), 1), [n, 1])
y = tf.tile(tf.expand_dims(tf.tile(tf.range(w), [h]), 1), [n, 1])
ind = tf.concat(1, [tf.expand_dims(repeat(tf.range(n), h*w), 1), x, y])

ht = 6
wt = 5

theta = tf.placeholder(tf.int32, [n, 2])
tx, ty = tf.split(1, 2, theta)
tx = tf.clip_by_value(tx, 0, ht-h)
ty = tf.clip_by_value(ty, 0, wt-w)

xt = x + tf.reshape(tf.tile(tx, [1,h*w]), [-1,1])
yt = y + tf.reshape(tf.tile(ty, [1,h*w]), [-1,1])
tind = tf.concat(1, [tf.expand_dims(repeat(tf.range(n), h*w), 1), xt, yt])

val = tf.ones(n*h*w)
T = tf.sparse_to_dense(tind, [n, ht, wt], val)

f = tf.reduce_sum(tf.square(T))
train_op = tf.train.AdamOptimizer().minimize(f)

sess = tf.Session()
print sess.run(T, {theta:np.array([[0, 1],[1, 3]])})
sess.run(train_op)
