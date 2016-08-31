import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchmat_to_tileimg
from attention import *
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/draw',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 10,
        """number of epochs to run""")
tf.app.flags.DEFINE_integer('n_hid', 300,
        """number of hidden units""")
tf.app.flags.DEFINE_integer('n_lat', 20,
        """number of latent variables""")
tf.app.flags.DEFINE_integer('N', 5,
        """attention size""")
tf.app.flags.DEFINE_integer('n_glim', 10,
        """number of glimpses""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

n_hid = FLAGS.n_hid
n_lat = FLAGS.n_lat
N = FLAGS.N
T = FLAGS.n_glim
height = 28
width = 28
n_in = height*width

LSTM = tf.nn.rnn_cell.BasicLSTMCell
attunit = AttentionUnit(height, width, 1, N)
RNN_enc = LSTM(n_hid, state_is_tuple=True)
RNN_dec = LSTM(n_hid, state_is_tuple=True)

x = tf.placeholder(tf.float32, [None, n_in])
hid_dec = tf.zeros([tf.shape(x)[0], n_hid])
state_enc = RNN_enc.zero_state(tf.shape(x)[0], tf.float32)
state_dec = RNN_dec.zero_state(tf.shape(x)[0], tf.float32)
p = [0]*T

x_err = x - tf.sigmoid(tf.zeros_like(x))
att_enc = linear(hid_dec, 5, scope='att_enc')
r = attunit.read([x, x_err], att_enc)
with tf.variable_scope('RNN_enc'):
    hid_enc, state_enc = RNN_enc(tf.concat(1, [r, hid_dec]), state_enc)
z_mean = linear(hid_enc, n_lat, scope='z_mean')
z_log_var = linear(hid_enc, n_lat, scope='z_log_var')
z = gaussian_sample(z_mean, z_log_var)
with tf.variable_scope('RNN_dec'):
    hid_dec, state_dec = RNN_dec(z, state_dec)
att_dec = linear(hid_dec, 5, scope='att_dec')
w = linear(hid_dec, attunit.read_dim, scope='w')
c = attunit.write(w, att_dec)
p[0] = tf.nn.sigmoid(c)
kld = gaussian_kld(z_mean, z_log_var, reduce_mean=False)

for t in range(1, T):
    x_err = x - tf.nn.sigmoid(c)
    att_enc = linear(hid_dec, 5, scope='att_enc', reuse=True)
    r = attunit.read([x, x_err], att_enc)
    with tf.variable_scope('RNN_enc', reuse=True):
        hid_enc, state_enc = RNN_enc(tf.concat(1, [r, hid_dec]), state_enc)
    z_mean = linear(hid_enc, n_lat, scope='z_mean', reuse=True)
    z_log_var = linear(hid_enc, n_lat, scope='z_log_var', reuse=True)
    z = gaussian_sample(z_mean, z_log_var)
    with tf.variable_scope('RNN_dec', reuse=True):
        hid_dec, state_dec = RNN_dec(z, state_dec)
    att_dec = linear(hid_dec, 5, scope='att_dec', reuse=True)
    w = linear(hid_dec, attunit.read_dim, scope='w', reuse=True)
    c = c + attunit.write(w, att_dec)
    p[t] = tf.nn.sigmoid(c)
    kld = kld + gaussian_kld(z_mean, z_log_var, reduce_mean=False)

neg_ll = bernoulli_neg_ll(x, p[-1])
kld = tf.reduce_mean(kld)
loss = neg_ll + kld
train_op = tf.train.AdamOptimizer().minimize(loss)

mnist = input_data.read_data_sets("data/mnist")
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_valid_batches = mnist.validation.num_examples / batch_size

saver = tf.train.Saver()
sess = tf.Session()

def train():
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    logfile.write(('n_in: %d, n_hid: %d, n_lat: %d\n' % (n_in, n_hid, n_lat)))
    sess.run(tf.initialize_all_variables())
    for i in range(FLAGS.n_epochs):
        start = time.time()
        train_neg_ll = 0.
        train_kld = 0.
        for j in range(n_train_batches):
            batch_x, _ = mnist.train.next_batch(batch_size)
            _, batch_neg_ll, batch_kld = \
                    sess.run([train_op, neg_ll, kld], {x:batch_x})
            train_neg_ll += batch_neg_ll
            train_kld += batch_kld
        train_neg_ll /= n_train_batches
        train_kld /= n_train_batches

        valid_neg_ll = 0.
        valid_kld = 0.
        for j in range(n_valid_batches):
            batch_x, _ = mnist.validation.next_batch(batch_size)
            batch_neg_ll, batch_kld = sess.run([neg_ll, kld], {x:batch_x})
            valid_neg_ll += batch_neg_ll
            valid_kld += batch_kld
        valid_neg_ll /= n_valid_batches
        valid_kld /= n_valid_batches

        line = "Epoch %d (%f sec), train loss %f = %f + %f, valid loss %f = %f + %f" \
                % (i+1, time.time()-start,
                        train_neg_ll+train_kld, train_neg_ll, train_kld,
                        valid_neg_ll+valid_kld, valid_neg_ll, valid_kld)
        print line
        logfile.write(line + '\n')
    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')
    batch_x, _ = mnist.test.next_batch(10)
    batch_p = sess.run(p, {x:batch_x})
    P = np.zeros((0, n_in))
    for i in range(10):
        P = np.concatenate([P, batch_x[i].reshape((1, n_in))], 0)
        for t in range(T):
            P = np.concatenate([P, batch_p[t][i].reshape((1, n_in))], 0)
    fig = plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(P, (28, 28), (10, T+1)))
    fig.savefig(FLAGS.save_dir+'/reconstructed.png')
    plt.show()


    """
    fig = plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(batch_x, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/original.png')

    fig = plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    p_recon = sess.run(p, {x:batch_x})
    plt.imshow(batchmat_to_tileimg(p_recon, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/reconstructed.png')

    p_gen = sess.run(p, {z:np.random.normal(size=(100, n_lat))})
    I_gen = batchmat_to_tileimg(p_gen, (height, width), (10, 10))
    fig = plt.figure('generated')
    plt.gray()
    plt.axis('off')
    plt.imshow(I_gen)
    fig.savefig(FLAGS.save_dir+'/generated.png')
    """

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
