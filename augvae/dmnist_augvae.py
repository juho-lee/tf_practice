import tensorflow as tf
import numpy as np
from utils.nn import *
from utils.image import batchmat_to_tileimg, batchimg_to_tileimg
from utils.misc import Logger
import time
import os
import matplotlib.pyplot as plt
from templates import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', './dmnist_results',
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

# image height=wid
wid = 56
n_cls = 14
# attended height=wid
awid = wid/2
smax = 0.4
feat_dim = 40
z_dim = 20

x = tf.placeholder(tf.float32, [None, wid**2])
x_img = tf.reshape(x, [-1, wid, wid, 1])
y = tf.placeholder(tf.int32, [None])
y_one_hot = one_hot(y, n_cls)

x_att0, x_att1 = dmnist_localize(x_img, awid, smax)
logits, cls0, cls1, feat0, feat1 = dmnist_classify(x_att0, x_att1, feat_dim, n_cls)
cent, acc = get_classification_loss(logits, y_one_hot)


# variational autoencoder
from utils.distribution import *
px = Bernoulli()
qz = Gaussian()
vae_is_tr = tf.placeholder(tf.bool)

# autoencode first feature
z0, kld0 = dmnist_encode(x_att0, z_dim, qz, vae_is_tr)
recon0, px_param0 = dmnist_decode(z0, feat0, px, vae_is_tr)
neg_ll0 = -px.log_likel(flat(x_att0), px_param0)

# autoencode second feature
z1, kld1 = dmnist_encode(x_att1, z_dim, qz, vae_is_tr, reuse=True)
recon1, px_param1 = dmnist_decode(z1, feat1, px, vae_is_tr, reuse=True)
neg_ll1 = -px.log_likel(flat(x_att1), px_param1)

neg_ll = 0.5*(neg_ll0 + neg_ll1)
kld = 0.5*(kld0 + kld1)

# discriminative regularization
recon0 = tf.reshape(recon0, [-1, awid, awid, 1])
recon1 = tf.reshape(recon1, [-1, awid, awid, 1])
rlogits, rcls0, rcls1, rfeat0, rfeat1 = \
        dmnist_classify(recon0, recon1, feat_dim, n_cls, reuse=True)
rcent, racc = get_classification_loss(rlogits, y_one_hot)
def l2norm(a, b):
    return tf.reduce_mean(tf.reduce_sum(tf.square(a-b), 1))
dreg = l2norm(rfeat0, feat0) + l2norm(rfeat1, feat1) + \
        l2norm(rlogits, logits) + rcent

# augmentative regularization
nz0 = tf.placeholder(tf.float32, [None, z_dim])
nz1 = tf.placeholder(tf.float32, [None, z_dim])
aug0, _ = dmnist_decode(nz0, feat0, px, vae_is_tr, reuse=True)
aug0 = tf.reshape(aug0, [-1, awid, awid, 1])
aug1, _ = dmnist_decode(nz1, feat1, px, vae_is_tr, reuse=True)
aug1 = tf.reshape(aug1, [-1, awid, awid, 1])
alogits, acls0, acls1, afeat0, afeat1 = \
        dmnist_classify(aug0, aug1, feat_dim, n_cls, reuse=True)
acent, aacc = get_classification_loss(alogits, y_one_hot)
areg = l2norm(afeat0, feat0) + l2norm(afeat1, feat1) + \
        l2norm(alogits, logits) + acent

sess = tf.Session()
all_vars = tf.all_variables()
cls_vars = [var for var in all_vars if (('loc' in var.name) | ('cls' in var.name))]
vae_vars = [var for var in all_vars if (('enc' in var.name) | ('dec' in var.name))]

train_cls = get_train_op(cent, var_list=cls_vars)
train_vae = get_train_op(neg_ll+kld+dreg+0.1*areg, var_list=vae_vars)
cls_saver = tf.train.Saver(cls_vars)
vae_saver = tf.train.Saver(vae_vars)

def cls_train():
    train_Logger = Logger('train cent', 'train acc')
    test_Logger = Logger('test acc')
    logfile = open(FLAGS.save_dir + '/cls_train.log', 'w', 0)
    idx = range(len(train_x))
    sess.run(tf.initialize_all_variables())
    for i in range(FLAGS.n_epochs):
        train_Logger.clear()
        np.random.shuffle(idx)
        start = time.time()
        for j in range(n_train_batches):
            batch_idx = idx[j*batch_size:(j+1)*batch_size]
            feed_dict = {x:train_x[batch_idx], y:train_y[batch_idx]}
            train_Logger.accum(sess.run([train_cls, cent, acc], feed_dict))

        test_Logger.clear()
        for j in range(n_test_batches):
            batch_idx = range(j*batch_size, (j+1)*batch_size)
            feed_dict = {x:test_x[batch_idx], y:test_y[batch_idx]}
            test_Logger.accum(sess.run(acc, feed_dict))

        line = train_Logger.get_status(i+1, time.time()-start) + \
                test_Logger.get_status_no_header()
        print line
        logfile.write(line + '\n')
    logfile.close()
    cls_saver.save(sess, FLAGS.save_dir+'/cls_model.ckpt')

def vae_train():
    train_Logger = Logger('train neg_ll', 'train kld', 'train areg', 'train aacc')
    test_Logger = Logger('test neg_ll', 'test kld', 'test areg', 'test aacc')
    logfile = open(FLAGS.save_dir + '/vae_train.log', 'w', 0)
    idx = range(len(train_x))
    sess.run(tf.initialize_all_variables())
    cls_saver.restore(sess, FLAGS.save_dir+'/cls_model.ckpt')
    vae_saver.restore(sess, FLAGS.save_dir+'/vae_model.ckpt')
    for i in range(FLAGS.n_epochs):
        train_Logger.clear()
        np.random.shuffle(idx)
        start = time.time()
        for j in range(n_train_batches):
            batch_idx = idx[j*batch_size:(j+1)*batch_size]
            feed_dict = {x:train_x[batch_idx], y:train_y[batch_idx], 
                    nz0:np.random.normal(size=(batch_size, z_dim)),
                    nz1:np.random.normal(size=(batch_size, z_dim)),
                    vae_is_tr:True}
            train_Logger.accum(sess.run([train_vae, neg_ll, kld, areg, aacc], feed_dict))

        test_Logger.clear()
        for j in range(n_test_batches):
            batch_idx = range(j*batch_size, (j+1)*batch_size)
            feed_dict = {x:test_x[batch_idx], y:test_y[batch_idx], 
                    nz0:np.random.normal(size=(batch_size, z_dim)),
                    nz1:np.random.normal(size=(batch_size, z_dim)),
                    vae_is_tr:False}
            test_Logger.accum(sess.run([neg_ll, kld, areg, aacc], feed_dict))

        line = train_Logger.get_status(i+1, time.time()-start) + \
                test_Logger.get_status_no_header()
        print line
        logfile.write(line + '\n')
    logfile.close()
    vae_saver.save(sess, FLAGS.save_dir+'/vae_model.ckpt')

def visualize():
    cls_saver.restore(sess, FLAGS.save_dir+'/cls_model.ckpt')
    vae_saver.restore(sess, FLAGS.save_dir+'/vae_model.ckpt')

    aug_acc_val = 0.
    for i in range(n_test_batches):
        batch_idx = range(i*batch_size, (i+1)*batch_size)
        feed_dict = {x:test_x[batch_idx], y:test_y[batch_idx],
                nz0:qz.prior_sample((batch_size, z_dim)),
                nz1:qz.prior_sample((batch_size, z_dim)),
                vae_is_tr:False}
        aug_acc_val += sess.run(aacc, feed_dict)
    print aug_acc_val/n_test_batches
    
    batch_feat = sess.run(feat0, {x:test_x[10:20]})    
    batch_feat = np.repeat(batch_feat, 10, 0)

    batch_aug0 = sess.run(aug0, {feat0:batch_feat, 
        nz0:qz.prior_sample((batch_size, z_dim)),
        nz1:qz.prior_sample((batch_size, z_dim)),
        vae_is_tr:False})
    plt.figure()
    plt.gray()
    plt.imshow(batchimg_to_tileimg(batch_aug0, (10, 10)))
    

    plt.show()
    
def one_shot_augmentation():
    cls_saver.restore(sess, FLAGS.save_dir+'/cls_model.ckpt')
    vae_saver.restore(sess, FLAGS.save_dir+'/vae_model.ckpt')

    n_one_shot_cls = 5
    n_examples_per_cls = 1
    n_aug_per_cls = 500

    # reserve for test
    n_test = 1000
    ind = range(len(one_shot_x))
    np.random.shuffle(ind)
    ote_x = one_shot_x[0:n_test]
    ote_y = one_shot_y[0:n_test] - 14
    otr_x = one_shot_x[n_test:]
    otr_y = one_shot_y[n_test:] - 14

    # sample one shot examples
    examples = [0]*n_one_shot_cls
    for i in range(n_one_shot_cls):
        ind = np.where(otr_y == i)[0]
        np.random.shuffle(ind)
        ind = ind[0:n_examples_per_cls]
        examples[i] = otr_x[ind]

    # augment data
    n_aug = n_one_shot_cls*n_aug_per_cls
    aug_x0 = np.zeros((n_aug, awid**2))
    aug_x1 = np.zeros((n_aug, awid**2))
    aug_y = np.repeat(range(n_one_shot_cls), n_aug_per_cls)
    for i in range(n_one_shot_cls):
        # extract features from one shot example
        ft0, ft1 = sess.run([feat0, feat1], {x:examples[i]})
        ft0 = np.repeat(ft0, n_aug_per_cls/n_examples_per_cls, 0)
        ft1 = np.repeat(ft1, n_aug_per_cls/n_examples_per_cls, 0)
        
        # augment
        ax0, ax1 = sess.run([flat(aug0), flat(aug1)],
                {feat0:ft0, feat1:ft1,
                    nz0:qz.prior_sample((n_aug_per_cls, z_dim)),
                    nz1:qz.prior_sample((n_aug_per_cls, z_dim)),
                    vae_is_tr:False})
        idx = range(i*n_aug_per_cls, (i+1)*n_aug_per_cls)
        aug_x0[idx] = ax0
        aug_x1[idx] = ax1

    plt.figure('aug0')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(aug_x0[2400:2500], (awid, awid), (10, 10)))
    plt.figure('aug1')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(aug_x1[2400:2500], (awid, awid), (10, 10)))
    plt.show()

    # save augmented data
    import cPickle as pkl
    import gzip
    filename = 'data/dmnist/dmnist_aug_' + str(n_examples_per_cls) + '.pkl.gz'
    f = gzip.open(filename, 'wb')
    pkl.dump([(aug_x0, aug_x1, aug_y), (otr_x, otr_y), (ote_x, ote_y)],
            f, protocol=2)
    f.close()

def one_shot_learning():
    # load data
    n_examples_per_cls = 5
    filename = 'data/dmnist/dmnist_aug_' + str(n_examples_per_cls) + '.pkl.gz'
    aug, tr, te = load_pkl(filename)
    aug_x0, aug_x1, aug_y = aug
    n_aug_batches = len(aug_x0)/batch_size
    otr_x, otr_y = tr
    n_otr_batches = len(otr_x)/batch_size
    ote_x, ote_y = te
    n_ote_batches = len(ote_x)/batch_size

    n_one_shot_cls = 5
    ox = tf.placeholder(tf.float32, [None, wid**2])
    ox_img = tf.reshape(ox, [-1, wid, wid, 1])
    oy = tf.placeholder(tf.int32, [None])
    oy_one_hot = one_hot(oy, n_one_shot_cls)

    # classify with real images    
    ox_att0, ox_att1 = dmnist_localize(ox_img, awid, smax, reuse=True)

    logits, _, _, _, _ = dmnist_classify(ox_att0, ox_att1, feat_dim, n_one_shot_cls,
            scope='ocls')
    cent, acc = get_classification_loss(logits, oy_one_hot)
    train_ocls = get_train_op(cent)

    # classify with augmented images
    ax_att0 = tf.placeholder(tf.float32, [None, awid**2])
    ax_att1 = tf.placeholder(tf.float32, [None, awid**2])
    alogits, _, _, _, _ = dmnist_classify(
            tf.reshape(ax_att0, [-1, awid, awid, 1]),
            tf.reshape(ax_att1, [-1, awid, awid, 1]),
            feat_dim, n_one_shot_cls,
            scope='acls')
    acent, aacc = get_classification_loss(alogits, oy_one_hot)
    train_acls = get_train_op(acent)
    # for test
    alogits_test, _, _, _, _ = dmnist_classify(ox_att0, ox_att1, feat_dim, n_one_shot_cls,
            scope='acls', reuse=True)
    _, aacc_test = get_classification_loss(alogits_test, oy_one_hot)

    sess.run(tf.initialize_all_variables())
    cls_saver.restore(sess, FLAGS.save_dir + '/cls_model.ckpt')
    
    train_Logger = Logger('train cent', 'train acc')
    test_Logger = Logger('test acc')

    """
    idx = range(len(otr_x))
    for i in range(FLAGS.n_epochs):
        train_Logger.clear()
        np.random.shuffle(idx)
        start = time.time()
        for j in range(n_otr_batches):
            batch_idx = idx[j*batch_size:(j+1)*batch_size]            
            feed_dict = {ox:otr_x[batch_idx], oy:otr_y[batch_idx]}
            train_Logger.accum(sess.run([train_ocls, cent, acc], feed_dict))

        test_Logger.clear()
        for j in range(n_ote_batches):
            batch_idx = range(j*batch_size, (j+1)*batch_size)
            feed_dict = {ox:ote_x[batch_idx], oy:ote_y[batch_idx]}
            test_Logger.accum(sess.run(acc, feed_dict))

        line = train_Logger.get_status(i+1, time.time()-start) + \
                test_Logger.get_status_no_header()
        print line
    """

    print 
    idx = range(len(aug_x0))
    for i in range(FLAGS.n_epochs):
        train_Logger.clear()
        np.random.shuffle(idx)
        start = time.time()
        for j in range(n_aug_batches):
            batch_idx = idx[j*batch_size:(j+1)*batch_size]            
            feed_dict = {ax_att0:aug_x0[batch_idx], 
                    ax_att1:aug_x1[batch_idx],
                    oy:aug_y[batch_idx]}
            train_Logger.accum(sess.run([train_acls, acent, aacc], feed_dict))

        test_Logger.clear()
        for j in range(n_ote_batches):
            batch_idx = range(j*batch_size, (j+1)*batch_size)
            feed_dict = {ox:ote_x[batch_idx], oy:ote_y[batch_idx]}
            test_Logger.accum(sess.run(aacc_test, feed_dict))

        line = train_Logger.get_status(i+1, time.time()-start) + \
                test_Logger.get_status_no_header()
        print line
    
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
