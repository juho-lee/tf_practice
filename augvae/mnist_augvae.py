import tensorflow as tf
import numpy as np
from utils.nn import *
from utils.image import batchmat_to_tileimg, batchimg_to_tileimg
from utils.misc import Logger
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', './mnist_results',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 5,
        """number of epochs to run""")
tf.app.flags.DEFINE_integer('mode', 0,
        """0: train classifier, 1: train VAE, 2: visualize""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_test_batches = mnist.test.num_examples / batch_size

wid = 28
n_cls = 10
x = tf.placeholder(tf.float32, [None, wid**2])
x_img = tf.reshape(x, [-1, wid, wid, 1])
y_one_hot = tf.placeholder(tf.float32, [None, n_cls])

def classify(img, reuse=None):
    with tf.variable_scope('cls', reuse=reuse):
        net = pool(conv(x_img, 16, 3), 2)
        net_out = [flat(net)]
        net = pool(conv(x_img, 32, 3), 2)
        net_out += [flat(net)]
        net = pool(conv(x_img, 64, 3), 2)
        net_out += [flat(net)]
        net = fc(flat(net), 256)
        logits = linear(net, n_cls)
        return logits, net_out

logits, net_out = classify(x_img)
cent, acc = get_classification_loss(logits, y_one_hot)

from utils.distribution import *
px = Bernoulli()
qz = Gaussian()
z_dim = 40
vae_is_tr = tf.placeholder(tf.bool)
def autoencode(x, reuse=None):
    with tf.variable_scope('vae', reuse=reuse):
        hid = fc_bn(x, 400, vae_is_tr)
        hid = linear(hid, 2*z_dim)
        qz_param = qz.get_param(hid)
        z = qz.sample(qz_param)
        hid = fc_bn(z, 400, vae_is_tr)
        hid = linear(hid, wid**2)
        px_param = px.get_param(hid)
        neg_ll = -px.log_likel(x, px_param)
        kld = qz.kld(qz_param)
        return z, px_param, neg_ll, kld
z, px_param, neg_ll, kld = autoencode(x)
recon = px.mean(px_param)
rlogits, rnet_out = classify(recon, reuse=True)
rcent, racc = get_classification_loss(rlogits, y_one_hot)

sess = tf.Session()
all_vars = tf.all_variables()
cls_vars = [var for var in all_vars if 'cls' in var.name]
vae_vars = [var for var in all_vars if 'vae' in var.name]
cls_saver = tf.train.Saver(cls_vars)
vae_saver = tf.train.Saver(vae_vars)
train_cls = get_train_op(cent, var_list=cls_vars)
train_vae = get_train_op(neg_ll+kld, var_list=vae_vars)

def cls_train():
    train_Logger = Logger('train cent', 'train acc')
    test_Logger = Logger('test acc')
    logfile = open(FLAGS.save_dir + '/cls_train.log', 'w', 0)
    sess.run(tf.initialize_all_variables())
    for i in range(FLAGS.n_epochs):
        train_Logger.clear()
        start = time.time()
        for j in range(n_train_batches):
            batch_x, batch_y_one_hot = mnist.train.next_batch(batch_size)
            feed_dict = {x:batch_x, y_one_hot:batch_y_one_hot}
            train_Logger.accum(sess.run([train_cls, cent, acc], feed_dict))

        test_Logger.clear()
        for j in range(n_test_batches):
            batch_x, batch_y_one_hot = mnist.test.next_batch(batch_size)
            feed_dict = {x:batch_x, y_one_hot:batch_y_one_hot}
            test_Logger.accum(sess.run(acc, feed_dict))

        line = train_Logger.get_status(i+1, time.time()-start) + \
                test_Logger.get_status_no_header()
        print line
        logfile.write(line + '\n')
    logfile.close()
    cls_saver.save(sess, FLAGS.save_dir+'/cls_model.ckpt')

def vae_train():
    train_Logger = Logger('train neg_ll', 'train kld', 'train rcent', 'train racc')
    test_Logger = Logger('test neg_ll', 'test kld', 'test rcent', 'test racc')
    logfile = open(FLAGS.save_dir + '/vae_train.log', 'w', 0)
    sess.run(tf.initialize_all_variables())
    cls_saver.restore(sess, FLAGS.save_dir+'/cls_model.ckpt')
    for i in range(FLAGS.n_epochs):
        train_Logger.clear()
        start = time.time()
        for j in range(n_train_batches):
            batch_x, batch_y_one_hot = mnist.train.next_batch(batch_size)
            feed_dict = {x:batch_x, y_one_hot:batch_y_one_hot, vae_is_tr:True}
            train_Logger.accum(sess.run([train_vae, neg_ll, kld, rcent, racc], feed_dict))

        test_Logger.clear()
        for j in range(n_test_batches):
            batch_x, batch_y_one_hot = mnist.test.next_batch(batch_size)
            feed_dict = {x:batch_x, y_one_hot:batch_y_one_hot, vae_is_tr:False}
            test_Logger.accum(sess.run([neg_ll, kld, rcent, racc], feed_dict))

        line = train_Logger.get_status(i+1, time.time()-start) + \
                test_Logger.get_status_no_header()
        print line
        logfile.write(line + '\n')
    logfile.close()
    vae_saver.save(sess, FLAGS.save_dir+'/vae_model.ckpt')

def visualize():
    raise NotImplementedError

def main(argv=None):
    if FLAGS.mode == 0:
        cls_train()
    elif FLAGS.mode == 1:
        vae_train()
    elif FLAGS.mode == 2:
        visualize()
    else:
        print "Unknown mode option"
        return

if __name__ == '__main__':
    tf.app.run()
