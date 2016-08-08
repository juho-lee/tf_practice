import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from attention import *

LSTM = tf.nn.rnn_cell.BasicLSTMCell

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

    def get_output(self, x):
        batch_size = tf.shape(x)[0]
        c = [0]*self.n_step
        z_mean = [0]*self.n_step
        z_log_var = [0]*self.n_step
        h_dec = tf.zeros((batch_size, self.n_hid))
        cell_enc = self.enc_RNN.zero_state(batch_size, tf.float32)
        cell_dec = self.dec_RNN.zero_state(batch_size, tf.float32)

        # t == 0
        x_err = x - tf.sigmoid(tf.zeros(tf.shape(x)))
        r = self.attunit.read(x, x_err, h_dec)
        with tf.variable_scope("enc_RNN"):
            h_enc, cell_enc = self.enc_RNN(cell_enc, tf.concat(1, [r,h_dec]))
        z_mean[0] = linear(h_enc, self.n_lat, "z_mean")
        z_log_var[0] = linear(h_enc, self.n_lat, "z_log_var")
        z = gaussian_sample(z_mean[0], z_log_var[0])
        with tf.variable_scope("dec_RNN"):
            h_dec, cell_dec = self.dec_RNN(cell_dec, z)
        c[0] = self.attunit.write(h_dec)
        # kld loss
        L_z = gaussian_kld(z_mean[0], z_log_var[0])

        # t > 0
        for t in range(1, self.n_step):
            x_err = x - tf.sigmoid(c[t-1])
            r = self.attunit.read(x, x_err, h_dec, reuse=True)
            with tf.variable_scope("enc_RNN", reuse=True):
                h_enc, cell_enc = self.enc_RNN(cell_enc, tf.concat(1, [r,h_dec]))
            z_mean[t] = linear(h_enc, self.n_lat, "z_mean", reuse=True)
            z_log_var[t] = linear(h_enc, self.n_lat, "z_log_var", reuse=True)
            z = gaussian_sample(z_mean[t], z_log_var[t])
            with tf.variable_scope("dec_RNN", reuse=True):
                h_dec, cell_dec = self.dec_RNN(cell_dec, z)
            c[t] = c[t-1] + write(h_dec)
            L_z = L_z + gaussian_kld(z_mean[t], z_log_var[t])

        dec_x = tf.nn.sigmoid(c[-1])
        L_x = bernoulli_ll_loss(x, dec_x)
        return L_x, L_z, dec_x, z, c

image_shape = [1, 28, 28]
N = 5
n_hid = 200
n_lat = 20
n_step = 10
model = DRAW(image_shape, N, n_hid, n_lat, n_step)

x = tf.placeholder(tf.float32, [None, 784])
L_x, L_z, dec_x, z, c = model.get_output(x)
