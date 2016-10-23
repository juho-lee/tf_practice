import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils.nn import *
from utils.image import batchimg_to_tileimg
from utils.misc import Logger
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/catgan',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 5,
        """number of epochs to run""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

# leaky relu
def lrelu(x, leak=0.1):
    f1 = 0.5*(1 + leak)
    f2 = 0.5*(1 - leak)
    return tf.nn.relu(x)

height = 28
width = 28
n_cls = 10
z_dim = 128

is_training = tf.placeholder(tf.bool)
def discriminate(x, reuse=None):
    with tf.variable_scope('Dis', reuse=reuse):
        out = conv(x, 32, 5, activation_fn=lrelu)
        out = pool(out, 3, stride=2)
        out = conv_bn(out, 64, 3, is_training, activation_fn=lrelu)
        out = conv_bn(out, 64, 3, is_training, activation_fn=lrelu)
        out = pool(out, 3, stride=2)
        out = conv_bn(out, 128, 3, is_training, activation_fn=lrelu)
        out = conv_bn(out, 10, 1, is_training, activation_fn=lrelu)
        out = fc_bn(flat(out), 128, is_training)
        out = fc(out, n_cls, activation_fn=tf.nn.softmax)

        return out

def generate(z, reuse=None):
    with tf.variable_scope('Gen', reuse=reuse):
        out = fc(z, 4*4*128)
        out = tf.reshape(out, [-1, 4, 4, 128])
        out = deconv_bn(out, 96, 3, is_training, stride=2, activation_fn=lrelu)
        out = deconv_bn(out, 64, 5, is_training, stride=2, activation_fn=lrelu)
        out = deconv_bn(out, 64, 5, is_training, stride=2, activation_fn=lrelu)
        out = conv(out, 1, 5, padding='VALID', activation_fn=tf.nn.sigmoid)
      
        return out

x = tf.placeholder(tf.float32, [None, height*width])
y = tf.placeholder(tf.int32, [None])
y_one_hot = one_hot(y, n_cls)
z = tf.placeholder(tf.float32, [None, z_dim])
x_img = tf.reshape(x, [-1,height,width,1])

# discriminator output for real image
p_real = discriminate(x_img)

# fake image generated from generator
fake = generate(z)
# discriminator output for fake image
p_fake = discriminate(fake, reuse=True)
batch_p_fake = tf.reduce_mean(p_fake, 0, keep_dims=True)

# losses
def cross_entropy(p, y, tol=1e-10):
    log_p = tf.log(p + tol)
    return -tf.reduce_mean(tf.reduce_sum(y*log_p, 1))

def entropy(p, tol=1e-10):
    log_p = tf.log(p + tol)
    return -tf.reduce_mean(tf.reduce_sum(p*log_p, 1))

L_D = cross_entropy(p_real, y_one_hot) - entropy(p_fake)
L_G = -entropy(batch_p_fake) + entropy(p_fake)

# get train ops
learning_rate = tf.placeholder(tf.float32)
vars = tf.trainable_variables()
D_vars = [var for var in vars if 'Dis' in var.name]
G_vars = [var for var in vars if 'Gen' in var.name]

train_D = get_train_op(L_D, var_list=D_vars,
        learning_rate=learning_rate, grad_clip=10.)

train_G = get_train_op(L_G, var_list=G_vars,
        learning_rate=learning_rate, grad_clip=10.)

# to check classification accuracy
correct = tf.equal(tf.argmax(p_real, 1), tf.argmax(y_one_hot, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

# load data
mnist = input_data.read_data_sets('data/mnist')
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_test_batches = mnist.test.num_examples / batch_size

saver = tf.train.Saver()
sess = tf.Session()
def train():
    train_Logger = Logger('train L_D', 'train acc', 'train L_G')
    test_Logger = Logger('test L_D', 'test acc', 'test L_G')
    
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    sess.run(tf.initialize_all_variables())
    lr_D = 0.0001
    lr_G = 0.001
    start = time.time()
    for i in range(FLAGS.n_epochs):
        train_Logger.clear()
        start = time.time()
        for j in range(n_train_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # train discriminator
            batch_z = np.random.normal(size=(batch_size, z_dim))
            feed_dict = {x:batch_x, y:batch_y, z:batch_z,
                    learning_rate:lr_D, is_training:True}
            D_res = sess.run([train_D, L_D, acc], feed_dict)

            # train generator
            batch_z = np.random.normal(size=(batch_size, z_dim))
            feed_dict = {x:batch_x, y:batch_y, z:batch_z,
                    learning_rate:lr_G, is_training:True}
            G_res = sess.run([train_G, L_G], feed_dict)

            train_Logger.accum(D_res + G_res)

            if (j+1)%50 == 0:
                print train_Logger.get_status(i+1, time.time()-start, it=j+1)

        test_Logger.clear()
        for j in range(n_test_batches):
            batch_x, batch_y = mnist.test.next_batch(batch_size)
            batch_z = np.random.normal(size=(batch_size, z_dim))
            feed_dict = {x:batch_x, y:batch_y, z:batch_z, is_training:False}
            test_Logger.accum(sess.run([L_D, acc, L_G], feed_dict))

        line = train_Logger.get_status(i+1, time.time()-start) + \
                test_Logger.get_status_no_header() + '\n'
        print line
        logfile.write(line + '\n')

    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')
    fig = plt.figure('generated')
    gen = sess.run(fake, {z:np.random.normal(size=(100, z_dim)), is_training:False})
    plt.gray()
    plt.axis('off')
    plt.imshow(batchimg_to_tileimg(gen, (10, 10)))
    fig.savefig(FLAGS.save_dir+'/genereated.png')

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
