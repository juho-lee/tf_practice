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

x = tf.placeholder(tf.float32, shape=[None, n_in])
is_training = tf.placeholder(tf.bool)
hid_enc0 = fc_bn(x, n_hid, is_training)
att_enc0 = to_att(hid_enc0)

ratio = 1.0
x_att0 = attunit.read(x, att_enc0, delta_max=ratio)
hid_enc0 = fc(tf.concat(1, [att_enc0, x_att0]), n_hid)
z0_mean = linear(hid_enc0, n_lat/2)
z0_log_var = linear(hid_enc0, n_lat/2)
z0 = gaussian_sample(z0_mean, z0_log_var)
hid_dec0 = fc_bn(z0, n_hid, is_training)
att_dec0 = to_att(hid_dec0)
c_att0 = linear(hid_dec0, N*N)
c0 = attunit.write(c_att0, att_dec0, delta_min=1.0/ratio)

res = tf.nn.relu(x - tf.nn.sigmoid(c0))
hid_enc1 = fc_bn(res, n_hid, is_training)
att_enc1 = to_att(hid_enc1)
x_att1 = attunit.read(res, att_enc1, delta_max=ratio)
hid_enc1 = fc(tf.concat(1, [att_enc1, x_att1]), n_hid)
z1_mean = linear(hid_enc1, n_lat/2)
z1_log_var = linear(hid_enc1, n_lat/2)
z1 = gaussian_sample(z1_mean, z1_log_var)
hid_dec1 = fc_bn(z1, n_hid, is_training)
att_dec1 = to_att(hid_dec1)
c_att1 = linear(hid_dec1, N*N)
c1 = attunit.write(c_att1, att_dec1, delta_min=1.0/ratio)

p = tf.nn.sigmoid(c0 + c1)

neg_ll = bernoulli_neg_ll(x, p)
kld = gaussian_kld(z0_mean, z0_log_var) + gaussian_kld(z1_mean, z1_log_var)
loss = neg_ll + kld

train_x, valid_x, test_x = load_pkl('data/dmnist/dmnist.pkl.gz')
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
                    sess.run([train_op, neg_ll, kld], {x:batch_x, is_training:True})
            train_neg_ll += batch_neg_ll
            train_kld += batch_kld
        train_neg_ll /= n_train_batches
        train_kld /= n_train_batches

        valid_neg_ll = 0.
        valid_kld = 0.
        for j in range(n_valid_batches):
            batch_x = valid_x[j*batch_size:(j+1)*batch_size]
            batch_neg_ll, batch_kld = sess.run([neg_ll, kld],
                    {x:batch_x, is_training:False})
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

    fa, sa = sess.run([tf.nn.sigmoid(x_att0), tf.nn.sigmoid(x_att1)],
            {x:batch_x, is_training:False})
    plt.figure('first att')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(fa, (N, N), (10, 10)))

    plt.figure('second att')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(sa, (N, N), (10, 10)))

    fa, sa = sess.run([tf.nn.sigmoid(c0), tf.nn.sigmoid(c1)],
            {x:batch_x, is_training:False})
    plt.figure('first canv')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(fa, (height, width), (10, 10)))

    plt.figure('second canv')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(sa, (height, width), (10, 10)))


    """
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
