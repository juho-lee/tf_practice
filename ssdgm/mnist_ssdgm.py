import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchmat_to_tileimg
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/ssdgm',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 20,
        """number of epochs to run""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

height = 28
width = 28
n_in = height*width

x = tf.placeholder(tf.float32, [None, n_in])
y = tf.placeholder(tf.float32, [None, 10])
is_train = tf.placeholder(tf.bool)

hid = tf.nn.relu(batch_norm(linear(x, 500)+linear(y, 500), is_train))
hid = fc_bn(hid, 500, is_train)
q_z_mean = linear(hid, 50)
q_z_log_var = linear(hid, 50)
z = gaussian_sample(q_z_mean, q_z_log_var)
hid = tf.nn.relu(batch_norm(linear(z, 500)+linear(y, 500), is_train))
hid = fc_bn(hid, 500, is_train)
p_x = fc(hid, n_in, activation_fn=tf.nn.sigmoid)

"""
hid = tf.nn.relu(linear(x, 500) + linear(y, 500))
q_z_mean = linear(hid, 50)
q_z_log_var = linear(hid, 50)
z = gaussian_sample(q_z_mean, q_z_log_var)
hid = tf.nn.relu(linear(z, 500) + linear(y, 500))
p_x = fc(hid, n_in, activation_fn=tf.nn.sigmoid)
"""

loss = bernoulli_neg_ll(x, p_x) + gaussian_kld(q_z_mean, q_z_log_var)

mnist = input_data.read_data_sets("data/mnist", one_hot=True)
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_valid_batches = mnist.validation.num_examples / batch_size

train_op = get_train_op(loss, grad_clip=10.)
saver = tf.train.Saver()
sess = tf.Session()

def train():
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    sess.run(tf.initialize_all_variables())
    for i in range(FLAGS.n_epochs):
        start = time.time()
        train_loss = 0.
        for j in range(n_train_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, batch_loss = sess.run([train_op, loss],
                    {x:batch_x, y:batch_y, is_train:True})
            train_loss += batch_loss
        train_loss /= n_train_batches

        valid_loss = 0.
        for j in range(n_valid_batches):
            batch_x, batch_y = mnist.validation.next_batch(batch_size)
            batch_loss = sess.run(loss, {x:batch_x, y:batch_y, is_train:False})
            valid_loss += batch_loss
        valid_loss /= n_valid_batches

        line = "Epoch %d (%f sec), train loss %f, valid loss %f" \
                % (i+1, time.time()-start, train_loss, valid_loss)
        print line
        logfile.write(line + '\n')
    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')
    batch_x, batch_y = mnist.test.next_batch(100)

    """
    fig = plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(batch_x, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/original.png')

    fig = plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    p_recon = sess.run(p_x, {x:batch_x, y:batch_y})
    plt.imshow(batchmat_to_tileimg(p_recon, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/reconstructed.png')
    """

    batch_z = np.random.normal(size=(100, 50))
    batch_y = np.zeros((100, 10))
    for i in range(10):
        batch_y[i*10:(i+1)*10, i] = 1.0
    fig = plt.figure('generated')
    plt.gray()
    plt.axis('off')
    p_gen = sess.run(p_x, {z:batch_z, y:batch_y, is_train:False})
    plt.imshow(batchmat_to_tileimg(p_gen, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/generated.png')

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
