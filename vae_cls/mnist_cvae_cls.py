import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchmat_to_tileimg, batchimg_to_tileimg
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/cvae',
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
x_img = tf.reshape(x, [-1, height, width, 1])
hid_enc = conv(x_img, n_ch, [ksize, ksize], [2, 2])
hid_enc = conv(hid_enc, n_ch*2, [ksize, ksize], [2, 2])
hid_enc = conv(hid_enc, n_ch*4, [ksize, ksize], [2, 2], padding='VALID')
hid_enc = flat(hid_enc)
z_mean = linear(hid_enc, n_lat)
z_log_var = linear(hid_enc, n_lat)
z = gaussian_sample(z_mean, z_log_var)
hid_dec = fc(z, n_ch*4*3*3)
hid_dec = tf.reshape(hid_dec, [-1, 3, 3, n_ch*4])
hid_dec = deconv(hid_dec, n_ch*2, [3, 3], [2, 2], padding='VALID')
hid_dec = deconv(hid_dec, n_ch, [2, 2], [2, 2])
p = deconv(hid_dec, 1, [2, 2], [2, 2], activation_fn=tf.nn.sigmoid)

y = tf.placeholder(tf.int32, [None])
y_one_hot = tf.contrib.layers.one_hot_encoding(y, 10)
cl = pool(conv(p, 16, [3, 3]), [2, 2])
cl = pool(conv(cl, 32, [3, 3]), [2, 2])
cl = pool(conv(cl, 64, [3, 3]), [2, 2])
cl = fc(flat(cl), 256)
y_logits = fc(cl, 10, activation_fn=None)
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y_logits, y_one_hot))
kld = gaussian_kld(z_mean, z_log_var)
correct = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))

loss = cross_entropy + kld + bernoulli_neg_ll(x, flat(p))
train_op = get_train_op(loss)

mnist = input_data.read_data_sets("data/mnist")
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_valid_batches = mnist.validation.num_examples / batch_size
n_test_batches = mnist.test.num_examples / batch_size

saver = tf.train.Saver()
sess = tf.Session()

def train():
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    sess.run(tf.initialize_all_variables())
    for i in range(FLAGS.n_epochs):
        start = time.time()
        train_acc = 0.
        train_ce = 0.
        train_kld = 0.
        for j in range(n_train_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, batch_acc, batch_ce, batch_kld \
                    = sess.run([train_op, accuracy, cross_entropy, kld],
                            {x:batch_x, y:batch_y})
            train_acc += batch_acc
            train_ce += batch_ce
            train_kld += batch_kld
        train_acc /= n_train_batches
        train_ce /= n_train_batches
        train_kld /= n_train_batches

        valid_acc = 0.
        valid_ce = 0.
        valid_kld = 0.
        for j in range(n_valid_batches):
            batch_x, batch_y = mnist.validation.next_batch(batch_size)
            batch_acc, batch_ce, batch_kld \
                    = sess.run([accuracy, cross_entropy, kld],
                            {x:batch_x, y:batch_y})
            valid_acc += batch_acc
            valid_ce += batch_ce
            valid_kld += batch_kld
        valid_acc /= n_valid_batches
        valid_ce /= n_valid_batches
        valid_kld /= n_valid_batches

        line = "Epoch %d (%f sec), train acc %f, ce %f, kld %f, valid acc %f, ce %f, kld %f" \
                % (i+1, time.time()-start, train_acc, train_ce, train_kld, valid_acc, valid_ce, valid_kld)
        print line
        logfile.write(line + '\n')
    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')

    test_acc = 0.
    for i in range(n_test_batches):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        batch_acc = sess.run(accuracy, {x:batch_x, y:batch_y})
        test_acc += batch_acc
    test_acc /= n_test_batches
    print 'test acc %f\n' % (test_acc)

    batch_x, batch_y = mnist.test.next_batch(100)
    fig = plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(batch_x, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/original.png')

    fig = plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    p_recon = sess.run(p, {x:batch_x})
    plt.imshow(batchimg_to_tileimg(p_recon, (10, 10)))
    fig.savefig(FLAGS.save_dir+'/reconstructed.png')

    p_gen = sess.run(p, {z:np.random.normal(size=(100, n_lat))})
    fig = plt.figure('generated')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchimg_to_tileimg(p_gen, (10, 10)))
    fig.savefig(FLAGS.save_dir+'/generated.png')

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
