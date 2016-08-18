import tensorflow as tf
from tensorflow.contrib import layers
from spatial_transformer import *
import numpy as np
fc = layers.fully_connected
conv = layers.convolution2d

x = tf.placeholder(tf.float32, [20, 10])
t = fc(x, 20)
tg = tf.gather(t, tf.cast(tf.linspace(0., 1., 10), 'int32'))
loss = tg * tg
train_step = tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run(train_step, {x:np.random.normal(size=(20, 10))})

"""
z = tf.placeholder(tf.float32, [None, 20])
t = tf.reshape(fc(z, 22), [-1, 1])
x = tf.linspace(0., 1., 20)*20
x = tf.cast(x, 'int32')
p = tf.gather(t, x)

loss = p*p
train_step = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run(train_step, {z:np.random.normal(size=(1, 20))})
    tt,xx = sess.run([t,x], {z:np.random.normal(size=(1,20))})
"""
