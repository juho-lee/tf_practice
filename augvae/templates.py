import tensorflow as tf
from utils.nn import *
from attention.spatial_transformer import *

def dmnist_localize(img, awid, smax, 
        scope='loc', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        loc = conv(pool(img, 2), 20, 3)
        loc = conv(pool(loc, 2), 20, 3)
        loc = fc(flat(loc), 200)
        hid = fc(loc, 200)
        theta0 = to_loc(hid, s_max=smax, is_simple=True)
        att0 = spatial_transformer(img, theta0, awid, awid)
        hid = fc(tf.concat(1, [hid, loc]), 200)
        theta1 = to_loc(hid, s_max=smax, is_simple=True)
        att1 = spatial_transformer(img, theta1, awid, awid)
        return att0, att1

def dmnist_classify(att0, att1, feat_dim, n_cls,
        scope='cls', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        cls0 = pool(conv(att0, 16, 3), 2)
        cls0 = pool(conv(cls0, 32, 3), 2)
        cls0 = flat(pool(conv(cls0, 64, 3), 2))
        
        cls1 = pool(conv(att1, 16, 3), 2)
        cls1 = pool(conv(cls1, 32, 3), 2)
        cls1 = flat(pool(conv(cls1, 64, 3), 2))

        # feature extraction
        with tf.variable_scope('feat', reuse=reuse):
            feat0 = fc(cls0, 200)
            feat0 = linear(feat0, feat_dim)
        with tf.variable_scope('feat', reuse=True):
            feat1 = fc(cls1, 200)
            feat1 = linear(feat1, feat_dim)
       
        # classification
        logits = fc(tf.concat(1, [feat0, feat1]), 400)
        logits = fc(logits, 200)
        logits = linear(logits, n_cls)
        return logits, cls0, cls1, feat0, feat1

def dmnist_encode(img, z_dim, qz, is_tr, 
        scope='enc', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        hid = conv_bn(img, 32, 5, is_tr, stride=2)
        hid = conv_bn(hid, 64, 5, is_tr, stride=2)
        hid = fc_bn(flat(hid), 1024, is_tr)
        hid = linear(hid, 2*z_dim)
        qz_param = qz.get_param(hid)
        z = qz.sample(qz_param)
        kld = qz.kld(qz_param)
        return z, kld

def dmnist_decode(z, feat, px, is_tr,
        scope='dec', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        hid = fc_bn(tf.concat(1, [z, feat]), 1024, is_tr)
        hid = fc_bn(tf.concat(1, [hid, feat]), 7*7*64, is_tr)
        hid = tf.reshape(hid, [-1, 7, 7, 64])
        hid = deconv_bn(hid, 32, 5, is_tr, stride=2)
        hid = deconv(hid, 1, 5, stride=2, activation_fn=None)
        px_param = px.get_param(flat(hid))
        recon = px.mean(px_param)        
        return recon, px_param

