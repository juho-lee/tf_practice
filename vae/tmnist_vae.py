import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import time
from utils.image import batchmat_to_tileimg
import matplotlib.pyplot as plt
fc = layers.fully_connected

def gaussian_sample(mean, log_var):
    eps = tf.random_normal(tf.shape(log_var))
    return mean + tf.mul(eps, tf.exp(0.5*log_var))

def gaussian_kld(mean, log_var):
    return tf.reduce_mean(
            -0.5*tf.reduce_sum(1+log_var-tf.pow(mean,2)-tf.exp(log_var), 1))

def bernoulli_neg_ll(x, p, eps=1.0e-10):
    return tf.reduce_mean(
            -tf.reduce_sum(x*tf.log(p+eps)+(1.-x)*tf.log(1-p+eps), 1))

h = 60
w = 60
n_hid = 300
n_lat = 20

x = tf.placeholder(tf.float32, [None, h*w])
h_enc = fc(x, n_hid)
z_mean = fc(h_enc, n_lat, activation_fn=None)
z_log_var = fc(h_enc, n_lat, activation_fn=None)
z = gaussian_sample(z_mean, z_log_var)

h_dec = fc(z, n_hid)
p = fc(h_dec, h*w, activation_fn=tf.nn.sigmoid)

neg_ll = bernoulli_neg_ll(x, p)
kld = gaussian_kld(z_mean, z_log_var)
loss = neg_ll + kld
train_step = tf.train.AdamOptimizer().minimize(loss)

from utils.data import load_pkl
from utils.image import mat_to_tileimg
import matplotlib.pyplot as plt

train_xy, _, test_xy = load_pkl('data/tmnist/tmnist.pkl.gz')
train_x, _ = train_xy
test_x, _ = test_xy

batch_size = 100
n_train_batches = len(train_x)/batch_size
n_test_batches = len(test_x)/batch_size

n_epochs = 20
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

    I_orig = batchmat_to_tileimg(test_x[0:batch_size], (60, 60), (10, 10))
    p_test = sess.run(p, feed_dict={x:test_x[0:batch_size]})
    I_recon = batchmat_to_tileimg(p_test, (60, 60), (10, 10))
    plt.figure()
    plt.gray()
    plt.axis('off')
    plt.imshow(I_orig)
    plt.figure()
    plt.imshow(I_recon)
    plt.show()
