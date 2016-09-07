import tensorflow as tf
from utils.prob import *
from utils.nn import *
from utils.image import batchmat_to_tileimg
from utils.data import load_pkl
from draw.attention import *
from stn.spatial_transformer import *
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/dmnist/vae_dattn',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 30,
        """number of epochs to run""")
tf.app.flags.DEFINE_integer('n_hid', 500,
        """number of hidden units""")
tf.app.flags.DEFINE_integer('n_lat', 20,
        """number of latent variables""")
tf.app.flags.DEFINE_integer('N', 30,
        """attention size""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

n_hid = FLAGS.n_hid
n_lat = FLAGS.n_lat
N = FLAGS.N
height = 56
width = 56
attunit = AttentionUnit(height, width, 1, N)
n_in = height*width

x = tf.placeholder(tf.float32, [None, n_in])
hid_enc = fc(x, n_hid)
att_enc = to_att(hid_enc, n_att=2)
x_att = attunit.read_multiple(x, att_enc, 2, delta_max=0.5)
hid_enc = fc(tf.concat(1, [att_enc, x_att]), n_hid)
z_mean = linear(hid_enc, n_lat)
z_log_var = linear(hid_enc, n_lat)
z = gaussian_sample(z_mean, z_log_var)

hid_dec = fc(z, n_hid)
att_dec = to_att(hid_dec, n_att=2)
p_att = linear(hid_dec, 2*N*N)
p = attunit.write_multiple(p_att, att_dec, 2, delta_min=1.5)
p = tf.nn.sigmoid(tf.slice(p, [0,0], [-1,n_in]) \
        + tf.slice(p, [0,attunit.read_dim], [-1, n_in]))


neg_ll = bernoulli_neg_ll(x, p)
kld = gaussian_kld(z_mean, z_log_var)
loss = neg_ll + kld

train_x, valid_x, test_x = load_pkl('data/dmnist/dmnist_fixed.pkl.gz')
batch_size = 100
n_train_batches = len(train_x) / batch_size
n_valid_batches = len(valid_x) / batch_size

train_op = tf.train.AdamOptimizer().minimize(loss)
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
            batch_x = train_x[j*batch_size:(j+1)*batch_size]
            _, batch_neg_ll, batch_kld = \
                    sess.run([train_op, neg_ll, kld], {x:batch_x})
            train_neg_ll += batch_neg_ll
            train_kld += batch_kld
        train_neg_ll /= n_train_batches
        train_kld /= n_train_batches

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
    batch_x = test_x[0:100]


    fig = plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(batch_x, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/original.png')

    x_att0 = tf.slice(x_att, [0,0], [-1,attunit.read_dim])
    x_att1 = tf.slice(x_att, [0,attunit.read_dim], [-1,attunit.read_dim])
    fa, sa = sess.run([tf.nn.sigmoid(x_att0), tf.nn.sigmoid(x_att1)], {x:batch_x})
    plt.figure('first')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(fa, (N, N), (10, 10)))

    plt.figure('second')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(sa, (N, N), (10, 10)))


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

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
