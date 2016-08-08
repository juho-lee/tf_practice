import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

def linear(x, dim, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        return layers.fully_connected(x, dim, activation_fn=None,
                weights_initializer=tf.random_normal_initializer())

class AttentionUnit(object):
    def __init__(self, image_shape, N):
        if len(image_shape) == 2:
            self.c = 1
            self.h = image_shape[0]
            self.w = image_shape[1]
        else:
            self.c = image_shape[0]
            self.h = image_shape[1]
            self.w = image_shape[2]
        self.N = N
        self.read_dim = self.c*self.N*self.N
        self.write_dim = self.c*self.h*self.w

    def get_att_params(self, att):
        g_x, g_y, log_var, log_delta, log_gamma = tf.split(1, 5, att)
        g_x = 0.5*(self.w+1)*(g_x+1)
        g_y = 0.5*(self.h+1)*(g_y+1)
        sigma = tf.exp(0.5*log_var)
        delta = (max(self.h,self.w)-1)*tf.exp(log_delta)/(self.N-1)
        gamma = tf.exp(log_gamma)
        return g_x, g_y, sigma, delta, gamma

    def get_filterbanks(self, g_x, g_y, sigma, delta):
        tol = 1.0e-5
        ind = tf.reshape(tf.cast(tf.range(self.N), tf.float32), [1,-1])
        ind = ind - self.N/2 - 0.5
        mu_x = tf.reshape(g_x + ind*delta, [-1,self.N,1])
        mu_y = tf.reshape(g_y + ind*delta, [-1,self.N,1])
        a = tf.reshape(tf.cast(tf.range(self.w), tf.float32), [1,1,-1])
        b = tf.reshape(tf.cast(tf.range(self.h), tf.float32), [1,1,-1])
        var = tf.reshape(tf.square(sigma), [-1,1,1])
        F_x = tf.exp(-tf.square((a-mu_x))/(2*var))
        F_x = F_x/(tol + tf.reduce_sum(F_x, 2, keep_dims=True))
        F_y = tf.exp(-tf.square((b-mu_y))/(2*var))
        F_y = F_y/(tol + tf.reduce_sum(F_y, 2, keep_dims=True))
        if self.c > 1:
            F_x = tf.tile(F_x, [self.c, 1, 1])
            F_y = tf.tile(F_y, [self.c, 1, 1])
        return F_x, F_y

    def read(self, x, x_err, hid, reuse=False):
        att = linear(hid, 5, "read", reuse=reuse)
        g_x, g_y, sigma, delta, gamma = self.get_att_params(att)
        F_x, F_y = self.get_filterbanks(g_x, g_y, sigma, delta)
        F_xt = tf.transpose(F_x, [0,2,1])
        x_att = tf.reshape(tf.batch_matmul(F_y,
            tf.batch_matmul(tf.reshape(x, [-1,self.h,self.w]), F_xt)),
            [-1,self.read_dim]) * tf.reshape(gamma, [-1,1])
        x_err_att = tf.reshape(tf.batch_matmul(F_y,
            tf.batch_matmul(tf.reshape(x_err, [-1,self.h,self.w]), F_xt)),
            [-1,self.read_dim]) * tf.reshape(gamma, [-1,1])
        return tf.concat(1, [x_att, x_err_att])

    def write(self, hid, reuse=False):
        w = linear(hid, self.read_dim, "write_w", reuse=reuse)
        att = linear(hid, 5, "write_att", reuse=reuse)
        g_x, g_y, sigma, delta, gamma = self.get_att_params(att)
        F_x, F_y = self.get_filterbanks(g_x, g_y, sigma, delta)
        w_att = tf.reshape(tf.batch_matmul(tf.transpose(F_y, [0,2,1]),
            tf.batch_matmul(tf.reshape(w,[-1,self.N,self.N]), F_x)),
            [-1,self.write_dim]) * tf.reshape(1./gamma, [-1,1])
        return w_att

"""
N = 10
c = 1
h = 40
w = 50
n_hid = 200
unit = AttentionUnit([c,h,w], N)
x_ = tf.placeholder(tf.float32, [None, c*h*w])
hid_ = tf.placeholder(tf.float32, [None, n_hid])
x_att_ = unit.read(x_, x_, hid_)
x_att_reuse_ = unit.read(x_, x_, hid_, reuse=True)
w_att_ = unit.write(hid_)
w_att_reuse_ = unit.write(hid_, reuse=True)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    x_att, w_att, x_att_re, w_att_re = sess.run([x_att_, w_att_, x_att_reuse_, w_att_reuse_],
            feed_dict={x_:np.random.rand(4, c*h*w), hid_:0.01*np.random.normal(size=(4, n_hid))})
    print x_att.shape
    print w_att.shape

    print x_att.sum(), (x_att-x_att_re).sum()
    print w_att.sum(), (w_att-w_att_re).sum()
"""
