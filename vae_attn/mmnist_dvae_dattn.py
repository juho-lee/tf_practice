import tensorflow as tf
from utils.prob import *
from utils.nn import *
from utils.image import batchmat_to_tileimg
from utils.data import load_pkl
from draw.attention import *
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mmnist/dvae_dattn',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 30,
        """number of epochs to run""")
tf.app.flags.DEFINE_integer('n_hid', 500,
        """number of hidden units""")
tf.app.flags.DEFINE_integer('n_lat_t', 10,
        """number of latent variables for attention""")
tf.app.flags.DEFINE_integer('n_lat_c', 10,
        """number of latent variables for number""")
tf.app.flags.DEFINE_integer('N', 28,
        """attention size""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

n_hid = FLAGS.n_hid
n_lat_t = FLAGS.n_lat_t
n_lat_c = FLAGS.n_lat_c
N = FLAGS.N
height = 60
width = 60
attunit = AttentionUnit(height, width, 1, N)
n_in = height*width
delta_max = 0.4
K = 4

x = tf.placeholder(tf.float32, [None, n_in])
GRU = tf.nn.rnn_cell.GRUCell
rnn_t = GRU(n_hid)
state_t = rnn_t.zero_state(tf.shape(x)[0], tf.float32)
hid_t_enc = tf.zeros([tf.shape(x)[0], n_hid])
rnn_c = GRU(n_hid)
state_c = rnn_c.zero_state(tf.shape(x)[0], tf.float32)
hid_c_enc = tf.zeros([tf.shape(x)[0], n_hid])
trans = [0]*K
x_att = [0]*K
x_hat = [0]*K
z_c = [0]*K
z_t = [0]*K
p_att = [0]*K
pk = [0]*K

with tf.variable_scope('rnn_t_enc'):
    hid_t_enc, state_t = rnn_t(tf.concat(1, [x, hid_t_enc]), state_t)
z_t_mean = linear(hid_t_enc, n_lat_t, scope='z_t_mean')
z_t_log_var = linear(hid_t_enc, n_lat_t, scope='z_t_log_var')
z_t[0] = gaussian_sample(z_t_mean, z_t_log_var)
trans[0] = to_att(fc(z_t[0], 10, scope='trans_fc'), scope='trans')
x_att[0] = attunit.read(x, trans[0], delta_max=delta_max)
with tf.variable_scope('rnn_c_enc'):
    hid_c_enc, state_c = rnn_c(tf.concat(1, [x_att[0], hid_c_enc]), state_c)
z_c_mean = linear(hid_c_enc, n_lat_c, scope='z_c_mean')
z_c_log_var = linear(hid_c_enc, n_lat_c, scope='z_c_log_var')
z_c[0] = gaussian_sample(z_c_mean, z_c_log_var)
kld = gaussian_kld(z_t_mean, z_t_log_var) + \
        gaussian_kld(z_c_mean, z_c_log_var)
x_hat[0] = x - tf.clip_by_value(attunit.write(x_att[0], trans[0],
    delta_max=delta_max), 0, 1)

for k in range(1, K):
    with tf.variable_scope('rnn_t_enc', reuse=True):
        hid_t_enc, state_t = rnn_t(tf.concat(1, [x_hat[k-1], hid_t_enc]), state_t)
    z_t_mean = linear(hid_t_enc, n_lat_t, scope='z_t_mean', reuse=True)
    z_t_log_var = linear(hid_t_enc, n_lat_t, scope='z_t_log_var', reuse=True)
    z_t[k] = gaussian_sample(z_t_mean, z_t_log_var)
    trans[k] = to_att(fc(z_t[k], 10, scope='trans_fc', reuse=True),
            scope='trans', reuse=True)
    x_att[k] = attunit.read(x, trans[k], delta_max=delta_max)
    with tf.variable_scope('rnn_c_enc', reuse=True):
        hid_c_enc, state_c = rnn_c(tf.concat(1, [x_att[k], hid_c_enc]), state_c)
    z_c_mean = linear(hid_c_enc, n_lat_c, scope='z_c_mean', reuse=True)
    z_c_log_var = linear(hid_c_enc, n_lat_c, scope='z_c_log_var', reuse=True)
    z_c[k] = gaussian_sample(z_c_mean, z_c_log_var)
    kld = kld + gaussian_kld(z_t_mean, z_t_log_var) \
            + gaussian_kld(z_c_mean, z_c_log_var)
    x_hat[k] = x_hat[k-1] - tf.clip_by_value(attunit.write(x_att[k], trans[k],
        delta_max=delta_max), 0, 1)

hid_c_dec = fc(z_c[0], n_hid, scope='hid_c_dec')
p_att[0] = fc(hid_c_dec, N*N, activation_fn=tf.nn.sigmoid, scope='p_att')
pk[0] = attunit.write(p_att[0], trans[0], delta_max=delta_max)
p = pk[0]
for k in range(1, K):
    hid_c_dec = fc(z_c[k], n_hid, scope='hid_c_dec', reuse=True)
    p_att[k] = fc(hid_c_dec, N*N, activation_fn=tf.nn.sigmoid,
            scope='p_att', reuse=True)
    pk[k] = attunit.write(p_att[k], trans[k], delta_max=delta_max)
    p = p + pk[k]
p = tf.clip_by_value(p, 0, 1)

neg_ll = bernoulli_neg_ll(x, p)
loss = neg_ll + kld

train_xy, valid_xy, test_xy = load_pkl('data/mmnist/mmnist.pkl.gz')
train_x, _ = train_xy
valid_x, _ = valid_xy
test_x, test_y = test_xy
batch_size = 100
n_train_batches = len(train_x) / batch_size
n_valid_batches = len(valid_x) / batch_size

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
            batch_x = train_x[j*batch_size:(j+1)*batch_size]
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
            batch_x = valid_x[j*batch_size:(j+1)*batch_size]
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
    np.random.shuffle(test_x)
    ind = ((test_y==3).nonzero())[0][0:10]
    batch_x = test_x[ind]
    batch_x_att, batch_x_hat, batch_pk, batch_p = \
            sess.run([x_att, x_hat, pk, p], {x:batch_x})

    A = np.zeros((0, N*N))
    for i in range(10):
        for k in range(K):
            A = np.concatenate([A, batch_x_att[k][i].reshape((1, N*N))], 0)
    fig = plt.figure('attended')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(A, (N, N), (10, K)))
    fig.savefig(FLAGS.save_dir+'/attended.png')

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

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
