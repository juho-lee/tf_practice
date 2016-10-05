import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchmat_to_tileimg
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/cvae_bn',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 20,
        """number of epochs to run""")
tf.app.flags.DEFINE_integer('n_ch', 16,
        """number of hidden channels to start""")
tf.app.flags.DEFINE_integer('ksize', 3,
        """convolution kernel size in encoder""")
tf.app.flags.DEFINE_integer('n_lat', 20,
        """number of latent variables""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

n_lat = FLAGS.n_lat
n_ch = FLAGS.n_ch
ksize = FLAGS.ksize
height = 28
width = 28
n_in = height*width
x = tf.placeholder(tf.float32, shape=[None, n_in])
is_train = tf.placeholder(tf.bool)
x_img = tf.reshape(x, [-1, height, width, 1])
hid_enc = conv_bn(x_img, n_ch, [ksize, ksize], is_train, stride=[2, 2],
        scope='conv_bn_0')
hid_enc = conv_bn(hid_enc, n_ch*2, [ksize, ksize], is_train, stride=[2, 2],
        scope='conv_bn_1')
hid_enc = conv_bn(hid_enc, n_ch*4, [ksize, ksize], is_train, stride=[2, 2],
        scope='conv_bn_2', padding='VALID')
hid_enc = flat(hid_enc)
z_mean = linear(hid_enc, n_lat)
z_log_var = linear(hid_enc, n_lat)
z = gaussian_sample(z_mean, z_log_var)
hid_dec = fc_bn(z, n_ch*4*3*3, is_train, scope='fc_bn_0')
hid_dec = tf.reshape(hid_dec, [-1, 3, 3, n_ch*4])
hid_dec = deconv_bn(hid_dec, n_ch*2, [3, 3], is_train, stride=[2, 2],
        scope='deconv_bn_0', padding='VALID')
hid_dec = deconv_bn(hid_dec, n_ch, [2, 2], is_train, stride=[2, 2],
        scope='deconv_bn_1')
p = flat(deconv(hid_dec, 1, [2, 2], [2, 2], activation_fn=tf.nn.sigmoid))

mnist = input_data.read_data_sets("data/mnist")
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_valid_batches = mnist.validation.num_examples / batch_size

neg_ll = bernoulli_neg_ll(x, p)
kld = gaussian_kld(z_mean, z_log_var)
loss = neg_ll + kld
train_op = tf.train.AdamOptimizer().minimize(loss)
saver = tf.train.Saver()
sess = tf.Session()

def train():
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    logfile.write(('n_in: %d, n_lat: %d\n' % (n_in, n_lat)))
    sess.run(tf.initialize_all_variables())
    for i in range(FLAGS.n_epochs):
        start = time.time()
        train_neg_ll = 0.
        train_kld = 0.
        for j in range(n_train_batches):
            batch_x, _ = mnist.train.next_batch(batch_size)
            _, batch_neg_ll, batch_kld = \
                    sess.run([train_op, neg_ll, kld], {x:batch_x, is_train:True})
            train_neg_ll += batch_neg_ll
            train_kld += batch_kld
        train_neg_ll /= n_train_batches
        train_kld /= n_train_batches

        valid_neg_ll = 0.
        valid_kld = 0.
        for j in range(n_valid_batches):
            batch_x, _ = mnist.validation.next_batch(batch_size)
            batch_neg_ll, batch_kld = sess.run([neg_ll, kld], {x:batch_x, is_train:False})
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
    batch_x, _ = mnist.test.next_batch(100)
    fig = plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(batch_x, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/original.png')

    fig = plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    p_recon = sess.run(p, {x:batch_x, is_train:False})
    plt.imshow(batchmat_to_tileimg(p_recon, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/reconstructed.png')

    p_gen = sess.run(p, {z:np.random.normal(size=(100, n_lat)), is_train:False})
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
