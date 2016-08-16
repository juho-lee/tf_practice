import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from attention import *
import time

LSTM = tf.nn.rnn_cell.BasicLSTMCell

def gaussian_sample(mean, log_var):
    eps = tf.random_normal(tf.shape(log_var))
    return mean + tf.mul(eps, tf.exp(0.5*log_var))

def gaussian_kld(mean, log_var):
    return 0.5*tf.reduce_sum(tf.square(mean)+tf.exp(log_var)-log_var-1, 1)

"""
def gaussian_kld(mean, log_var):
    return tf.reduce_mean(
            -0.5*tf.reduce_sum(1+log_var-tf.pow(mean,2)-tf.exp(log_var), 1))
"""

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
        self.enc_RNN = LSTM(n_hid, state_is_tuple=True)
        self.dec_RNN = LSTM(n_hid, state_is_tuple=True)

    def get_output(self, x):
        batch_size = tf.shape(x)[0]
        h_dec = tf.zeros((batch_size, self.n_hid))
        enc_state = self.enc_RNN.zero_state(batch_size, tf.float32)
        dec_state = self.dec_RNN.zero_state(batch_size, tf.float32)
        dec_x = [0]*self.n_step

        # t == 0
        x_err = x - tf.sigmoid(tf.zeros(tf.shape(x)))
        r = self.attunit.read(x, x_err, h_dec)
        with tf.variable_scope("enc_RNN"):
            h_enc, enc_state = self.enc_RNN(tf.concat(1, [r,h_dec]), enc_state)
        z_mean = linear(h_enc, self.n_lat, "z_mean")
        z_log_var = linear(h_enc, self.n_lat, "z_log_var")
        z = gaussian_sample(z_mean, z_log_var)
        with tf.variable_scope("dec_RNN"):
            h_dec, dec_state = self.dec_RNN(z, dec_state)
        c = self.attunit.write(h_dec)
        dec_x[0] = tf.nn.sigmoid(c)
        # kld loss
        L_z = gaussian_kld(z_mean, z_log_var)

        # t > 0
        for t in range(1, self.n_step):
            x_err = x - tf.sigmoid(c)
            r = self.attunit.read(x, x_err, h_dec, reuse=True)
            with tf.variable_scope("enc_RNN", reuse=True):
                h_enc, enc_state = self.enc_RNN(tf.concat(1, [r,h_dec]), enc_state)
            z_mean = linear(h_enc, self.n_lat, "z_mean", reuse=True)
            z_log_var = linear(h_enc, self.n_lat, "z_log_var", reuse=True)
            z = gaussian_sample(z_mean, z_log_var)
            with tf.variable_scope("dec_RNN", reuse=True):
                h_dec, dec_state = self.dec_RNN(z, dec_state)
            c = c + self.attunit.write(h_dec, reuse=True)
            dec_x[t] = tf.nn.sigmoid(c)
            L_z = L_z + gaussian_kld(z_mean, z_log_var)

        L_x = bernoulli_ll_loss(x, dec_x[-1])
        L_z = tf.reduce_mean(L_z)
        return L_x, L_z, dec_x

if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("../data/mnist")
    batch_size = 100
    n_train_batches = mnist.train.num_examples / batch_size
    n_test_batches = mnist.test.num_examples / batch_size

    image_shape = [1, 28, 28]
    N = 5
    n_in = np.prod(image_shape)
    n_hid = 256
    n_lat = 20
    n_step = 10
    model = DRAW(image_shape, N, n_hid, n_lat, n_step)
    x = tf.placeholder(tf.float32, [None, n_in])
    L_x, L_z, dec_x = model.get_output(x)
    loss = L_x + L_z
    train_step = tf.train.AdamOptimizer().minimize(loss)

    from utils.image import mat_to_tileimg
    import matplotlib.pyplot as plt
    n_epochs = 4
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(n_epochs):
            start = time.time()
            train_L_x = 0.
            train_L_z = 0.
            for j in range(n_train_batches):
                batch_x, _ = mnist.train.next_batch(batch_size)
                _, batch_L_x, batch_L_z = \
                    sess.run([train_step, L_x, L_z], feed_dict={x:batch_x})
                train_L_x += batch_L_x
                train_L_z += batch_L_z
            train_L_x /= n_train_batches
            train_L_z /= n_train_batches
            print "Epoch %d (%f sec), train L_x %f, train L_z %f" \
                    % (i+1, time.time()-start, train_L_x, train_L_z)

        test_x, _ = mnist.test.next_batch(10)
        test_dec_x = sess.run(dec_x, feed_dict={x:test_x})

        X = np.zeros((0, n_in))
        for i in range(10):
            for j in range(n_step):
                strip = test_dec_x[j][i].reshape((1, n_in))
                X = np.concatenate([X, strip], axis=0)
        I = mat_to_tileimg(X, (28, 28), (10, n_step))
        plt.figure()
        plt.gray()
        plt.axis('off')
        plt.imshow(I)
        plt.show()
