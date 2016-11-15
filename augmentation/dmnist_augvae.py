import tensorflow as tf
import numpy as np
from utils.nn import *
from utils.image import *
from utils.misc import Logger
import time
import os
import matplotlib.pyplot as plt
from dmnist_templates import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', './dmnist_augvae',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 10,
        """number of epochs to run""")
tf.app.flags.DEFINE_integer('mode', 0,
        """0: train classifier, 1: train VAE, 2: visualize""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

# load data
from utils.data import load_pkl
train_xy, test_xy, one_shot_xy = load_pkl('data/dmnist/dmnist.pkl.gz')
train_x, train_y = train_xy
test_x, test_y = test_xy
one_shot_x, one_shot_y = one_shot_xy
batch_size = 100
n_train_batches = len(train_x)/batch_size
n_test_batches = len(test_x)/batch_size

# some utility functions
def l2norm(a, b):
    return tf.reduce_mean(tf.reduce_sum(tf.square(a-b), 1))

def gen_noise(batch_size):
    return qz.prior_sample(batch_size)

wid = 56
n_cls = 14
awid = wid/2
s_max = 0.4
ft_dim = 40
z_dim = 20

x = tf.placeholder(tf.float32, [None, wid**2])
x_img = tf.reshape(x, [-1, wid, wid, 1])
y = tf.placeholder(tf.int32, [None])
y_oh = one_hot(y, n_cls)

cls_is_tr = tf.placeholder(tf.bool)
att0, att1 = localize(x_img, awid, s_max)
logits, ft0, ft1 = classify(att0, att1, ft_dim, n_cls, cls_is_tr)

vae_is_tr = tf.placeholder(tf.bool)
from utils.distribution import *
qz = Gaussian(z_dim)
px = Bernoulli()
z0, kld0 = encode(att0, z_dim, qz, vae_is_tr)
recon0, neg_ll0 = decode(att0, z0, ft0, px, vae_is_tr)
z1, kld1 = encode(att1, z_dim, qz, vae_is_tr, reuse=True)
recon1, neg_ll1 = decode(att1, z1, ft1, px, vae_is_tr, reuse=True)
neg_ll = 0.5*(neg_ll0 + neg_ll1)
kld = 0.5*(kld0 + kld1)


# discriminative regularization
dlogits, dft0, dft1 = classify(recon0, recon1, ft_dim, n_cls, cls_is_tr, reuse=True)
dcent, dacc = get_classification_loss(dlogits, y_oh)
dreg = l2norm(dft0, ft0) + l2norm(dft1, ft1) + l2norm(dlogits, logits) + dcent

# augmentation regularization
nz0 = tf.placeholder(tf.float32, [None, z_dim])
nz1 = tf.placeholder(tf.float32, [None, z_dim])
aug0, _ = decode(att0, nz0, ft0, px, vae_is_tr, reuse=True)
aug1, _ = decode(att1, nz1, ft1, px, vae_is_tr, reuse=True)
alogits, aft0, aft1 = classify(aug0, aug1, ft_dim,  n_cls, cls_is_tr, reuse=True)
acent, aacc = get_classification_loss(alogits, y_oh)
areg = l2norm(aft0, ft0) + l2norm(aft1, ft1) + l2norm(alogits, logits) + acent

all_vars = tf.all_variables()
cls_vars = [var for var in all_vars if 'loc' in var.name or 'cls' in var.name]
vae_vars = [var for var in all_vars if 'enc' in var.name or 'dec' in var.name]

sess = tf.Session()
cls_saver = tf.train.Saver(cls_vars)
vae_saver = tf.train.Saver(vae_vars)

def cls_train():
    cent, acc = get_classification_loss(logits, y_oh)
    train_op = get_train_op(cent, var_list=cls_vars)

    train_Logger = Logger('train cent', 'train acc')
    test_Logger = Logger('test cent', 'test acc')
    logfile = open(FLAGS.save_dir + '/cls_train.log', 'w', 0)
    idx = range(len(train_x))
    sess.run(tf.initialize_all_variables())
    for i in range(FLAGS.n_epochs):
        train_Logger.clear()
        np.random.shuffle(idx)
        start = time.time()
        for j in range(n_train_batches):
            batch_idx = idx[j*batch_size:(j+1)*batch_size]
            feed_dict = {x:train_x[batch_idx], y:train_y[batch_idx], cls_is_tr:True}
            train_Logger.accum(sess.run([train_op, cent, acc], feed_dict))

        test_Logger.clear()
        for j in range(n_test_batches):
            batch_idx = range(j*batch_size, (j+1)*batch_size)
            feed_dict = {x:test_x[batch_idx], y:test_y[batch_idx], cls_is_tr:False}
            test_Logger.accum(sess.run([cent, acc], feed_dict))

        line = train_Logger.get_status(i+1, time.time()-start) + \
                test_Logger.get_status_no_header()
        print line
        logfile.write(line + '\n')
    logfile.close()
    cls_saver.save(sess, FLAGS.save_dir+'/cls_model.ckpt')

def vae_train():
    lam = 0.1
    loss = neg_ll+kld + lam*(dreg+areg)
    train_op = get_train_op(loss, var_list=vae_vars, grad_clip=10.)

    train_Logger1 = Logger('train neg_ll', 'train kld')
    train_Logger2 = Logger('train dreg', 'train dacc', 'train areg', 'train aacc')
    test_Logger1 = Logger('test neg_ll', 'test kld')
    test_Logger2 = Logger('test dreg', 'test dacc', 'test areg', 'test aacc')
    logfile = open(FLAGS.save_dir + '/vae_train.log', 'w', 0)
    idx = range(len(train_x))
    sess.run(tf.initialize_all_variables())
    cls_saver.restore(sess, FLAGS.save_dir+'/cls_model.ckpt')
    for i in range(FLAGS.n_epochs):
        train_Logger1.clear()
        train_Logger2.clear()
        np.random.shuffle(idx)
        start = time.time()
        for j in range(n_train_batches):
            batch_idx = idx[j*batch_size:(j+1)*batch_size]
            feed_dict = {x:train_x[batch_idx], y:train_y[batch_idx],
                    nz0:gen_noise(batch_size), nz1:gen_noise(batch_size),
                    cls_is_tr:False, vae_is_tr:True}
            train_Logger1.accum(sess.run([train_op, neg_ll, kld], feed_dict))
            train_Logger2.accum(sess.run([dreg, dacc, areg, aacc], feed_dict))

        test_Logger1.clear()
        test_Logger2.clear()
        for j in range(n_test_batches):
            batch_idx = range(j*batch_size, (j+1)*batch_size)
            feed_dict = {x:test_x[batch_idx], y:test_y[batch_idx],
                    nz0:gen_noise(batch_size), nz1:gen_noise(batch_size),
                    cls_is_tr:False, vae_is_tr:False}
            test_Logger1.accum(sess.run([neg_ll, kld], feed_dict))
            test_Logger2.accum(sess.run([dreg, dacc, areg, aacc], feed_dict))

        et = time.time()-start
        line = train_Logger1.get_status(i+1, et)
        print line
        logfile.write(line+'\n')
        
        line = train_Logger2.get_status(i+1, et)
        print line
        logfile.write(line+'\n')
        
        line = test_Logger1.get_status(i+1, et)
        print line
        logfile.write(line+'\n')
        
        line = test_Logger2.get_status(i+1, et) + '\n'
        print line
        logfile.write(line+'\n')

    logfile.close()
    vae_saver.save(sess, FLAGS.save_dir+'/vae_model.ckpt')

def visualize():
    cls_saver.restore(sess, FLAGS.save_dir+'/cls_model.ckpt')
    vae_saver.restore(sess, FLAGS.save_dir+'/vae_model.ckpt')

    test_Logger = Logger('test dcent', 'test dacc', 'test acent', 'test aacc')
    for j in range(n_test_batches):
        batch_idx = range(j*batch_size, (j+1)*batch_size)
        feed_dict = {x:test_x[batch_idx], y:test_y[batch_idx], 
                nz0:gen_noise(batch_size), nz1:gen_noise(batch_size),
                cls_is_tr:False, vae_is_tr:False}
        test_Logger.accum(sess.run([dcent, dacc, acent, aacc], feed_dict))
    print test_Logger.get_status_no_header(nocomma=True)

    batt0, batt1 = sess.run([att0, att1], {x:test_x[0:100], cls_is_tr:False})
    fig = create_fig('att0')
    plt.imshow(batchimg_to_tileimg(batt0, (10, 10)))
    fig = create_fig('att1')
    plt.imshow(batchimg_to_tileimg(batt1, (10, 10)))

    brecon0, brecon1 = sess.run([recon0, recon1], 
            {x:test_x[0:100], cls_is_tr:False, vae_is_tr:False})
    fig = create_fig('recon0')
    plt.imshow(batchimg_to_tileimg(brecon0, (10, 10)))
    fig = create_fig('recon1')
    plt.imshow(batchimg_to_tileimg(brecon1, (10, 10)))

    bft0 = sess.run(ft0, {x:test_x[0:10], cls_is_tr:False})
    bft0 = np.repeat(bft0, 10, 0)
    baug0 = sess.run(aug0, {ft0:bft0, nz0:gen_noise(100), nz1:gen_noise(100),
        cls_is_tr:False, vae_is_tr:False})
    fig = create_fig('aug0')
    plt.imshow(batchimg_to_tileimg(baug0, (10, 10)))

    plt.show()

def main(argv=None):
    if FLAGS.mode == 0:
        cls_train()
    elif FLAGS.mode == 1:
        vae_train()
    elif FLAGS.mode == 2:
        visualize()
    elif FLAGS.mode == 3:
        one_shot_augmentation()
    elif FLAGS.mode == 4:
        one_shot_learning()
    else:
        print "Unknown mode option"
        return

if __name__ == '__main__':
    tf.app.run()
