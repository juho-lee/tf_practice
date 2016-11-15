import tensorflow as tf
from utils.nn import *
from attention.spatial_transformer import *

def localize(img, awid, s_max, scope='loc', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        loc = conv(pool(img, 2), 20, 3)
        loc = conv(pool(img, 2), 20, 3)
        loc = fc(flat(loc), 200)
        hid = fc(loc, 200)
        theta0 = to_loc(hid, s_max=s_max, is_simple=True)
        att0 = spatial_transformer(img, theta0, awid, awid)
        hid = fc(tf.concat(1, [hid, loc]), 200)
        theta1 = to_loc(hid, s_max=s_max, is_simple=True)
        att1 = spatial_transformer(img, theta1, awid, awid)
        return att0, att1

def classify(att0, att1, ft_dim, n_cls, is_tr, scope='cls', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # feature extraction
        with tf.variable_scope('fext', reuse=reuse):
            cls0 = pool(conv_bn(att0, 16, 3, is_tr), 2)
            cls0 = pool(conv_bn(cls0, 32, 3, is_tr), 2)
            cls0 = pool(conv_bn(cls0, 64, 3, is_tr), 2)
            cls0 = pool(conv_bn(cls0, 128, 3, is_tr), 2)
            ft0 = fc(flat(cls0), ft_dim)
        with tf.variable_scope('fext', reuse=True):
            cls1 = pool(conv_bn(att1, 16, 3, is_tr), 2)
            cls1 = pool(conv_bn(cls1, 32, 3, is_tr), 2)
            cls1 = pool(conv_bn(cls1, 64, 3, is_tr), 2)
            cls1 = pool(conv_bn(cls1, 128, 3, is_tr), 2)
            ft1 = fc(flat(cls1), ft_dim)
        # classification
        logits = fc_bn(tf.concat(1, [ft0, ft1]), 200, is_tr)
        logits = linear(logits, n_cls)
        return logits, ft0, ft1

def encode(img, z_dim, qz, is_tr, scope='enc', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        hid = conv_bn(img, 32, 5, is_tr, stride=2, scope='bn0', reuse=reuse)
        hid = conv_bn(hid, 64, 5, is_tr, stride=2, scope='bn1', reuse=reuse)
        hid = fc_bn(flat(hid), 1024, is_tr, scope='bn2', reuse=reuse)
        hid = linear(hid, 2*z_dim)
        qz_param = qz.get_param(hid)
        z = qz.sample(qz_param)
        kld = qz.kld(qz_param)
        return z, kld

def decode(img, z, ft, px, is_tr, scope='dec', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        hid = fc_bn(tf.concat(1, [z, ft]), 1024, is_tr, scope='bn0', reuse=reuse)
        hid = fc_bn(tf.concat(1, [hid, ft]), 7*7*64, is_tr, scope='bn1', reuse=reuse)
        hid = tf.reshape(hid, [-1, 7, 7, 64])
        hid = deconv_bn(hid, 32, 5, is_tr, stride=2, scope='bn2', reuse=reuse)
        hid = deconv(hid, 1, 5, stride=2, activation_fn=None)
        px_param = px.get_param(hid)
        recon = px.mean(px_param)
        neg_ll = -px.log_likel(img, px_param)
        return recon, neg_ll

def discriminate(att, is_tr, scope='dis', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = conv_bn(att, 32, 5, is_tr, stride=2, activation_fn=lrelu)
        out = conv_bn(out, 64, 5, is_tr, stride=2, activation_fn=lrelu)
        out = fc_bn(flat(out), 1024, is_tr, activation_fn=lrelu)
        rho = linear(out, 1)
        return rho, out

def recognize(out, c_dim, qc, is_tr, scope='rec', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = fc_bn(out, 128, is_tr, activation_fn=lrelu)
        out = linear(out, c_dim)
        return qc.get_param(out)

def generate(z, c, ft, is_tr, scope='gen', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = fc_bn(tf.concat(1, [z, c, ft]), 1024, is_tr)
        out = fc_bn(tf.concat(1, [out, ft]), 7*7*64, is_tr)
        out = tf.reshape(out, [-1, 7, 7, 64])
        out = deconv_bn(out, 32, 5, is_tr, stride=2)
        out = deconv(out, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
        return out
