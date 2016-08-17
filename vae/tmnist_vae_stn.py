import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import time
import matplotlib.pyplot as plt
from utils.data import load_pkl
from utils.image import batchmat_to_tileimg, batchimg_to_tileimg
from spatial_transformer import spatial_transformer

fc = layers.fully_connected
conv = layers.convolution2d
pool = layers.max_pool2d
flat = layers.flatten

def gaussian_sample(mean, log_var):
    eps = tf.random_normal(tf.shape(log_var))
    return mean + tf.mul(eps, tf.exp(0.5*log_var))

def gaussian_kld(mean, log_var):
    return tf.reduce_mean(
            -0.5*tf.reduce_sum(1+log_var-tf.pow(mean,2)-tf.exp(log_var), 1))

def bernoulli_neg_ll(x, p, eps=1.0e-10):
    return tf.reduce_mean(
            -tf.reduce_sum(x*tf.log(p+eps)+(1.-x)*tf.log(1-p+eps), 1))

n_hid = 500
n_lat = 20

x = tf.placeholder(tf.float32, [None, 60*60])
x_img = tf.reshape(x, [-1, 60, 60, 1])
h_enc = fc(x, n_hid)
# encoder localization net
loc_enc = fc(fc(h_enc, 50), 6, activation_fn=None,
        weights_initializer=tf.constant_initializer(np.zeros((50, 6))),
        biases_initializer=tf.constant_initializer(
            np.array([[1.,0,0],[0,1.,0]]).flatten()))
x_trans = flat(spatial_transformer(x_img, loc_enc, [15, 15]))
h_enc = fc(tf.concat(1, [x_trans, h_enc]), n_hid)
z_mean = fc(h_enc, n_lat, activation_fn=None)
z_log_var = fc(h_enc, n_lat, activation_fn=None)
z = gaussian_sample(z_mean, z_log_var)

h_dec = fc(z, n_hid)
p_trans = tf.reshape(fc(h_dec, 15*15, activation_fn=None), [-1, 15, 15, 1])
# decoder localization net
loc_dec = fc(fc(h_dec, 50), 6, activation_fn=None,
        weights_initializer=tf.constant_initializer(np.zeros((50, 6))),
        biases_initializer=tf.constant_initializer(
            np.array([[1.,0,0],[0,1.,0]]).flatten()))
p = tf.nn.sigmoid(flat(spatial_transformer(p_trans, loc_dec, [60, 60])))

neg_ll = bernoulli_neg_ll(x, p)
kld = gaussian_kld(z_mean, z_log_var)
loss = neg_ll + kld
train_step = tf.train.AdamOptimizer().minimize(loss)

train_xy, _, test_xy = load_pkl('data/tmnist/tmnist.pkl.gz')
train_x, _ = train_xy
test_x, _ = test_xy

batch_size = 100
n_train_batches = len(train_x)/batch_size
n_test_batches = len(test_x)/batch_size

n_epochs = 10
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(n_epochs):
        start = time.time()
        train_neg_ll = 0.
        train_kld = 0.
        for j in range(n_train_batches):
            batch_x = train_x[j*batch_size:(j+1)*batch_size]
            _, batch_neg_ll, batch_kld = sess.run([train_step, neg_ll, kld],
                    feed_dict={x:batch_x})
            train_neg_ll += batch_neg_ll
            train_kld += batch_kld
        train_neg_ll /= n_train_batches
        train_kld /= n_train_batches
        print "Epoch %d (%.3f sec), train neg ll %f, train kld %f" \
                % (i+1, time.time()-start, train_neg_ll, train_kld)
        np.random.shuffle(train_x)
