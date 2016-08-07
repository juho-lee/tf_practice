import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from attention import *

def linear(input, num_outputs):
    return layers.fully_connected(input, num_outputs,
            weights_initializer=tf.random_normal_initializer(),
            activation_fn=None)
LSTM = tf.rnn_cell.BasicLSTMCell

def gaussian_sample(mean, log_var):
    eps = tf.random_normal(tf.shape(log_var))
    return mean + tf.mul(eps, tf.exp(0.5*log_var))

def gaussian_kld(mean, log_var):
    return tf.reduce_mean(
            -0.5*tf.reduce_sum(1+log_var-tf.pow(mean,2)-tf.exp(log_var), 1))

def bernoulli_ll_loss(x, p, eps=1.0e-10):
    return tf.reduce_mean(
            -tf.reduce_sum(x*tf.log(p+eps)+(1.-x)*tf.log(1-p+eps), 1))

class DRAW(object):
    def __init__(self, image_shape, N, n_hid, n_lat, n_step):
        self.attunit = AttentionUnit(image_shape, N)
        self.n_in = np.prod(image_shape)
        self.n_hid = n_hid
        self.n_lat = n_lat
        self.n_step = n_step

        self.enc_RNN = LSTM(n_hid, 2*self.attunit.read_dim+n_hid)
        self.dec_RNN = LSTM(n_hid, n_lat)

    def encode(self, x):

        c_prev = tf.zeros(tf.shape(x))
        h_dec_prev = tf.zeros((tf.shape(x)[0], self.n_hid))
        with tf.variable_scope("encode"):
            x_err = x - tf.sigmoid(c_prev)
            linear(h_dec_prev, 5)


