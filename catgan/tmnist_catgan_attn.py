import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils.nn import *
from utils.image import batchimg_to_tileimg
from utils.misc import Logger
import time
import os
import matplotlib.pyplot as plt
#from attention.draw_attention import *
from attention.spatial_transformer import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/tmnist/catgan_attn',
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

height = 50
width = 50
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

with tf.variable_scope('Loc'):
    loc_param = conv(pool(x_img, 2), 20, 3)
    loc_param = conv(pool(loc_param, 2), 20, 3)
    loc_param = to_loc(loc_param, s_max=0.5)
x_img_attn = spatial_transformer(x_img, loc_param, 28, 28)

# discriminator output for real image
p_real = discriminate(x_img_attn)

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
Loc_vars = [var for var in vars if 'Loc' in var.name]
D_vars = [var for var in vars if 'Dis' in var.name]
G_vars = [var for var in vars if 'Gen' in var.name]

train_D = get_train_op(L_D, var_list=D_vars+Loc_vars,
        learning_rate=learning_rate, grad_clip=10.)

train_G = get_train_op(L_G, var_list=G_vars,
        learning_rate=learning_rate, grad_clip=10.)

# to check classification accuracy
correct = tf.equal(tf.argmax(p_real, 1), tf.argmax(y_one_hot, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

# load data
from utils.data import load_pkl
train_xy, _, test_xy = load_pkl('data/tmnist/tmnist.pkl.gz')
train_x, train_y = train_xy
test_x, test_y = test_xy
batch_size = 100
n_train_batches = len(train_x) / batch_size
n_test_batches = len(test_x) / batch_size

saver = tf.train.Saver()
sess = tf.Session()
def train():
    idx = range(len(train_x))
    train_Logger = Logger('train L_D', 'train acc', 'train L_G')
    test_Logger = Logger('test L_D', 'test acc', 'test L_G')
    
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    sess.run(tf.initialize_all_variables())
    lr_D = 0.0001
    lr_G = 0.002
    start = time.time()
    for i in range(FLAGS.n_epochs):
        np.random.shuffle(idx)
        train_Logger.clear()
        start = time.time()
        for j in range(n_train_batches):
            batch_idx = idx[j*batch_size:(j+1)*batch_size]
            batch_x = train_x[batch_idx]
            batch_y = train_y[batch_idx]

            # train discriminator
            batch_z = np.random.normal(size=(batch_size, z_dim))
            feed_dict = {x:batch_x, y:batch_y, z:batch_z,
                    learning_rate:lr_D, is_training:True}
            D_res = sess.run([train_D, L_D, acc], feed_dict)

            # train generator
            for k in range(2):
                batch_z = np.random.normal(size=(batch_size, z_dim))
                feed_dict = {x:batch_x, y:batch_y, z:batch_z,
                        learning_rate:lr_G, is_training:True}
                G_res = sess.run([train_G, L_G], feed_dict)
            train_Logger.accum(D_res + G_res)

            if (j+1)%50 == 0:
                print train_Logger.get_status(i+1, time.time()-start, it=j+1)

        test_Logger.clear()
        for j in range(n_test_batches):
            batch_idx = range(j*batch_size, (j+1)*batch_size)
            batch_x = test_x[batch_idx]
            batch_y = test_y[batch_idx]
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
    fig = plt.figure('attended')
    attn = sess.run(x_img_attn, {x:test_x[0:100], is_training:False})
    plt.gray()
    plt.axis('off')
    plt.imshow(batchimg_to_tileimg(attn, (10, 10)))
    
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
