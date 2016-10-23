import tensorflow as tf
import numpy as np
from utils.nn import *
from utils.distribution import *
from utils.misc import Logger
from utils.data import load_pkl
from utils.image import batchmat_to_tileimg
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/dmnist/cvae',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 30,
        """number of epochs to run""")
tf.app.flags.DEFINE_integer('n_hid', 500,
        """number of hidden units""")
tf.app.flags.DEFINE_integer('n_lat', 50,
        """number of latent variables""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

qz = Gaussian()
px = Bernoulli()

n_hid = FLAGS.n_hid
n_lat = FLAGS.n_lat
height = 56
width = 56
n_in = height*width
x = tf.placeholder(tf.float32, shape=[None, n_in])
is_training = tf.placeholder(tf.bool)

x_img = tf.reshape(x, [-1, height, width, 1])
hid_enc = conv_bn(x_img, 32, 5, is_training, stride=2)
hid_enc = conv_bn(hid_enc, 64, 5, is_training, stride=2)
hid_enc = conv_bn(hid_enc, 128, 5, is_training, stride=2)
hid_enc = fc_bn(flat(hid_enc), 1024, is_training)
hid_enc = linear(hid_enc, n_lat*2)
qz_param = qz.get_param(hid_enc)
z = qz.sample(qz_param)
hid_dec = fc_bn(z, 1024, is_training)
hid_dec = fc_bn(hid_dec, 7*7*128, is_training)
hid_dec = tf.reshape(hid_dec, [-1, 7, 7, 128])
hid_dec = deconv_bn(hid_dec, 64, 5, is_training, stride=2)
hid_dec = deconv_bn(hid_dec, 32, 5, is_training, stride=2)
hid_dec = flat(deconv(hid_dec, 1, 5, stride=2, activation_fn=None))
px_param = px.get_param(hid_dec)

neg_ll = -px.log_likel(x, px_param)
kld = qz.kld(qz_param)
loss = neg_ll + kld
train_op = get_train_op(loss)

train_xy, test_xy, _ = load_pkl('data/dmnist/dmnist.pkl.gz')
train_x, train_y = train_xy
test_x, test_y = test_xy
batch_size = 100
n_train_batches = len(train_x)/batch_size
n_test_batches = len(test_x)/batch_size

sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

idx = range(len(train_x))
def train():
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    sess.run(tf.initialize_all_variables())
    train_Logger = Logger('train loss', 'train neg_ll', 'train kld')
    test_Logger = Logger('test_loss', 'test neg_ll', 'test kld')
    for i in range(FLAGS.n_epochs):
        np.random.shuffle(idx)
        train_Logger.clear()
        start = time.time()
        for j in range(n_train_batches):
            feed_dict = {x:train_x[idx[j*batch_size:(j+1)*batch_size]],
                    is_training:True}
            train_Logger.accum(sess.run([train_op, loss, neg_ll, kld], feed_dict))

        test_Logger.clear()
        for j in range(n_test_batches):
            feed_dict = {x:test_x[j*batch_size:(j+1)*batch_size], 
                    is_training:False}
            test_Logger.accum(sess.run([loss, neg_ll, kld], feed_dict))

        line = train_Logger.get_status(i+1, time.time()-start) + \
                test_Logger.get_status_no_header()
        print line
        logfile.write(line + '\n')
    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')
    
def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')
    batch_x = test_x[0:100]
    fig = plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(batch_x, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/original.png')

    fig = plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    recon = sess.run(px.mean(px_param), {x:batch_x, is_training:False})
    plt.imshow(batchmat_to_tileimg(recon, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/reconstructed.png')

    gen = sess.run(px.mean(px_param), {z:np.random.normal(size=(100, n_lat)),
        is_training:False})
    I_gen = batchmat_to_tileimg(gen, (height, width), (10, 10))
    fig = plt.figure('generated')
    plt.gray()
    plt.axis('off')
    plt.imshow(I_gen)
    fig.savefig(FLAGS.save_dir+'/generated.png')

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
