import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchmat_to_tileimg
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/csvae',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 20,
        """number of epochs to run""")
tf.app.flags.DEFINE_integer('n_ch', 16,
        """number of hidden channels to start""")
tf.app.flags.DEFINE_integer('ksize', 3,
        """convolution kernel size in encoder""")
tf.app.flags.DEFINE_integer('n_lat', 10,
        """number of latent variables""")
tf.app.flags.DEFINE_integer('n_fac', 10,
        """number of factors""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

n_lat = FLAGS.n_lat
n_fac = FLAGS.n_fac
n_ch = FLAGS.n_ch
ksize = FLAGS.ksize
height = 28
width = 28
n_in = height*width
x = tf.placeholder(tf.float32, shape=[None, n_in])
x_img = tf.reshape(x, [-1, height, width, 1])
hid_enc = conv(x_img, n_ch, [ksize, ksize], [2, 2])
hid_enc = conv(hid_enc, n_ch*2, [ksize, ksize], [2, 2])
hid_enc = conv(hid_enc, n_ch*4, [ksize, ksize], [2, 2], padding='VALID')
hid_enc = flat(hid_enc)
z_mean = linear(hid_enc, n_lat)
z_log_var = linear(hid_enc, n_lat)
z = gaussian_sample(z_mean, z_log_var)
w_mean = linear(hid_enc, n_fac)
w_log_var = linear(hid_enc, n_fac)
w = rect_gaussian_sample(w_mean, w_log_var)

hid_dec = fc(z, n_ch*4*3*3)
hid_dec = tf.reshape(hid_dec, [-1, 3, 3, n_ch*4])
hid_dec = deconv(hid_dec, n_ch/2, [3, 3], [2, 2], padding='VALID')
hid_dec = deconv(hid_dec, n_ch/4, [2, 2], [2, 2])
p = tf.slice(w, [0,0], [-1,1]) * flat(deconv(hid_dec, 1, [2, 2], [2, 2], activation_fn=None))
for i in range(1, n_fac):
    hid_dec = fc(z, n_ch*4*3*3)
    hid_dec = tf.reshape(hid_dec, [-1, 3, 3, n_ch*4])
    hid_dec = deconv(hid_dec, n_ch/2, [3, 3], [2, 2], padding='VALID')
    hid_dec = deconv(hid_dec, n_ch/4, [2, 2], [2, 2])
    p = p + tf.slice(w, [0,i], [-1,1]) * flat(deconv(hid_dec, 1, [2, 2], [2, 2], activation_fn=None))
p = tf.nn.sigmoid(p)

#p = tf.reduce_sum(tf.expand_dims(tf.expand_dims(w,1), 1)*p, 3)
#p = tf.nn.sigmoid(flat(p))

mnist = input_data.read_data_sets("data/mnist")
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_valid_batches = mnist.validation.num_examples / batch_size

neg_ll = bernoulli_neg_ll(x, p)
kld = gaussian_kld(z_mean, z_log_var) + \
        rect_gaussian_kld(w_mean, w_log_var, mean0=-1.0)
loss = neg_ll + kld
train_op = get_train_op(loss)
saver = tf.train.Saver()
sess = tf.Session()

def train():
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    logfile.write(('n_in: %d, n_ch: %d, ksize: %d, n_lat: %d\n' % (n_in, n_ch, ksize, n_lat)))
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
    batch_x, _ = mnist.test.next_batch(batch_size)
    fig = plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(batch_x, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/original.png')

    fig = plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    p_recon, batch_w = sess.run([p, w], {x:batch_x})
    plt.imshow(batchmat_to_tileimg(p_recon, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/reconstructed.png')

    batch_w = np.zeros((n_fac*n_fac, n_fac))
    for i in range(n_fac):
        batch_w[i*n_fac:(i+1)*n_fac, i] = 1.0
    batch_z = np.random.normal(size=(n_fac*n_fac, n_lat))
    p_gen = sess.run(p, {w:batch_w, z:batch_z})
    I_gen = batchmat_to_tileimg(p_gen, (height, width), (n_fac, n_fac))
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
