import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchmat_to_tileimg
from utils.data import load_pkl
from draw.attention import *
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/svae_attn',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 30,
        """number of epochs to run""")
tf.app.flags.DEFINE_integer('n_hid', 300,
        """number of hidden units""")
tf.app.flags.DEFINE_integer('n_lat_t', 10,
        """number of latent variables for attention""")
tf.app.flags.DEFINE_integer('n_lat_c', 10,
        """number of latent variables for number""")
tf.app.flags.DEFINE_integer('N', 5,
        """attention size""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

n_hid = FLAGS.n_hid
n_lat_t = FLAGS.n_lat_t
n_lat_c = FLAGS.n_lat_c
N = FLAGS.N
height = 28
width = 28
attunit = AttentionUnit(height, width, 1, N)
n_in = height*width
delta_max = None
K = 5
x_att = [0]*K
p_att = [0]*K

x = tf.placeholder(tf.float32, [None, n_in])
hid_t_enc = fc(x, n_hid, scope='hid_t_enc')
#w_mean = linear(hid_t_enc, 1, scope='w_mean')
#w_log_var = linear(hid_t_enc, 1, scope='w_log_var')
#w = rect_gaussian_sample(w_mean, w_log_var)
z_t_mean = linear(hid_t_enc, n_lat_t, scope='z_t_mean')
z_t_log_var = linear(hid_t_enc, n_lat_t, scope='z_t_log_var')
z_t = gaussian_sample(z_t_mean, z_t_log_var)
trans = to_att(fc(z_t, 10, scope='trans_fc'), scope='trans')
x_att[0] = attunit.read(x, trans, delta_max=delta_max)
hid_c_enc = fc(x_att[0], n_hid, scope='hid_c_enc')
z_c_mean = linear(hid_c_enc, n_lat_c, scope='z_c_mean')
z_c_log_var = linear(hid_c_enc, n_lat_c, scope='z_c_log_var')
z_c = gaussian_sample(z_c_mean, z_c_log_var)
x_hat = x - tf.clip_by_value(
        attunit.write(x_att[0], trans, delta_max=delta_max), 0, 1)
hid_dec = fc(z_c, n_hid, scope='hid_dec')
p_att[0] = fc(hid_dec, N*N, activation_fn=tf.nn.sigmoid, scope='p_att')
p = tf.clip_by_value(
        attunit.write(p_att[0], trans, delta_max=delta_max), 0, 1)
kld = gaussian_kld(z_t_mean, z_t_log_var) + \
        gaussian_kld(z_c_mean, z_c_log_var)
     #   rect_gaussian_kld(w_mean, w_log_var)

for k in range(1, K):
    hid_t_enc = fc(x_hat, n_hid, scope='hid_t_enc', reuse=True)
    #w_mean = linear(hid_t_enc, 1, scope='w_mean', reuse=True)
    #w_log_var = linear(hid_t_enc, 1, scope='w_log_var', reuse=True)
    #w = rect_gaussian_sample(w_mean, w_log_var)
    z_t_mean = linear(hid_t_enc, n_lat_t, scope='z_t_mean', reuse=True)
    z_t_log_var = linear(hid_t_enc, n_lat_t, scope='z_t_log_var', reuse=True)
    z_t = gaussian_sample(z_t_mean, z_t_log_var)
    trans = to_att(fc(z_t, 10, scope='trans_fc', reuse=True),
            scope='trans', reuse=True)
    x_att[k] = attunit.read(x, trans, delta_max=delta_max)
    hid_c_enc = fc(x_att[k], n_hid, scope='hid_c_enc', reuse=True)
    z_c_mean = linear(hid_c_enc, n_lat_c, scope='z_c_mean', reuse=True)
    z_c_log_var = linear(hid_c_enc, n_lat_c, scope='z_c_log_var', reuse=True)
    z_c = gaussian_sample(z_c_mean, z_c_log_var)
    x_hat = x - tf.clip_by_value(
            attunit.write(x_att[k], trans, delta_max=delta_max), 0, 1)
    hid_dec = fc(z_c, n_hid, scope='hid_dec', reuse=True)
    p_att[k] = fc(hid_dec, N*N, activation_fn=tf.nn.sigmoid,
            scope='p_att', reuse=True)
    p = tf.clip_by_value(p +
            attunit.write(p_att[k], trans, delta_max=delta_max), 0, 1)
    kld = kld + gaussian_kld(z_t_mean, z_t_log_var) + \
            gaussian_kld(z_c_mean, z_c_log_var)
            # rect_gaussian_kld(w_mean, w_log_var)

neg_ll = bernoulli_neg_ll(x, p)
loss = neg_ll + kld

mnist = input_data.read_data_sets("data/mnist")
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_valid_batches = mnist.validation.num_examples / batch_size

learning_rate = tf.placeholder(tf.float32)
#train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
train_op = get_train_op(loss, learning_rate=learning_rate, grad_clip=10.)
saver = tf.train.Saver()
sess = tf.Session()

def train():
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    logfile.write(('n_in: %d, n_hid: %d, n_lat_t: %d, n_lat_c: %d\n' % (n_in, n_hid, n_lat_t, n_lat_c)))
    sess.run(tf.initialize_all_variables())
    lr = 0.001
    for i in range(FLAGS.n_epochs):
        start = time.time()
        train_neg_ll = 0.
        train_kld = 0.
        for j in range(n_train_batches):
            batch_x, _ = mnist.train.next_batch(batch_size)
            _, batch_neg_ll, batch_kld = \
                    sess.run([train_op, neg_ll, kld], {x:batch_x, learning_rate:lr})
            train_neg_ll += batch_neg_ll
            train_kld += batch_kld
        train_neg_ll /= n_train_batches
        train_kld /= n_train_batches
        if (i+1) % 3 == 0:
            lr = lr * 0.8

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
    batch_x_att, batch_p = sess.run([x_att, p], {x:batch_x})

    A = np.zeros((0, N*N))
    for i in range(10):
        for k in range(K):
            A = np.concatenate([A, batch_x_att[k][i].reshape((1, N*N))], 0)
    fig = plt.figure('attended')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(A, (N, N), (10, K)))
    fig.savefig(FLAGS.save_dir+'/attended.png')

    """
    P = np.zeros((0, n_in))
    for i in range(10):
        P = np.concatenate([P, batch_x[i].reshape((1, n_in))], 0)
        for k in range(K):
            P = np.concatenate([P, batch_pk[k][i].reshape((1, n_in))], 0)
        P = np.concatenate([P, batch_p[i].reshape((1, n_in))])
    fig = plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(P, (height, width), (10, K+2)))
    fig.savefig(FLAGS.save_dir+'/reconstructed.png')
    """

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
