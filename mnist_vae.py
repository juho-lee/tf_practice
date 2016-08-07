import tensorflow as tf
import tensorflow.contrib.layers as layers

def gaussian_sample(mean, log_var):
    eps = tf.random_normal(tf.shape(log_var))
    return mean + tf.mul(eps, tf.exp(0.5*log_var))

def gaussian_kld(mean, log_var):
    return tf.reduce_mean(
            -0.5*tf.reduce_sum(1+log_var-tf.pow(mean,2)-tf.exp(log_var), 1))

def bernoulli_ll_loss(x, p, eps=1.0e-10):
    return tf.reduce_mean(
            -tf.reduce_sum(x*tf.log(p+eps)+(1.-x)*tf.log(1-p+eps), 1))

class VAE(object):
    def __init__(self, n_in, n_hid, n_lat):
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_lat = n_lat

        self.x = tf.placeholder(tf.float32, shape=[None, n_in])
        self.enc_h = layers.fully_connected(self.x, n_hid)
        self.z_mean = layers.fully_connected(self.enc_h, n_lat,
                activation_fn=None)
        self.z_log_var = layers.fully_connected(self.enc_h, n_lat,
                activation_fn=None)
        self.z = gaussian_sample(self.z_mean, self.z_log_var)
        self.dec_h = layers.fully_connected(self.z, n_hid)
        self.dec_p = layers.fully_connected(self.dec_h, n_in,
                activation_fn=tf.nn.sigmoid)
    def get_ll_loss(self):
        return bernoulli_ll_loss(self.x, self.dec_p)

    def get_kld(self):
        return gaussian_kld(self.z_mean, self.z_log_var)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/mnist")
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_test_batches = mnist.test.num_examples / batch_size

n_in = 784
n_hid = 200
n_lat = 10
model = VAE(n_in, n_hid, n_lat)
ll_loss = model.get_ll_loss()
kld = model.get_kld()
loss = ll_loss + kld
train_step = tf.train.AdamOptimizer().minimize(loss)

n_epochs = 20
import time
from utils import mat_to_tileimg
from utils import gen_grid
import matplotlib.pyplot as plt
import numpy as np
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(n_epochs):
        start = time.time()
        train_ll_loss = 0.
        train_kld = 0.
        for j in range(n_train_batches):
            batch_x, _ = mnist.train.next_batch(batch_size)
            _, batch_ll_loss, batch_kld = \
                    sess.run([train_step, ll_loss, kld], feed_dict={model.x:batch_x})
            train_ll_loss += batch_ll_loss
            train_kld += batch_kld

        train_ll_loss /= n_train_batches
        train_kld /= n_train_batches
        print "Epoch %d (%f sec), train ll loss %f, train kld %f" \
                % (i+1, time.time()-start, train_ll_loss, train_kld)
    test_x, _ = mnist.test.next_batch(batch_size)
    I_orig = mat_to_tileimg(test_x, (28, 28), (10, 10))
    p_recon = sess.run(model.dec_p, feed_dict={model.x:test_x})
    I_recon = mat_to_tileimg(p_recon, (28, 28), (10, 10))
    eps = gen_grid(2, 10) if n_lat == 2 \
            else np.random.normal(size=(batch_size,n_lat))
    p_gen = sess.run(model.dec_p, feed_dict={model.z:eps})
    I_gen = mat_to_tileimg(p_gen, (28, 28), (10, 10))
    plt.figure()
    plt.gray()
    plt.axis('off')
    plt.imshow(I_gen)
    plt.figure()
    plt.imshow(I_orig)
    plt.figure()
    plt.imshow(I_recon)
    plt.show()
