import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchimg_to_tileimg
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/dmnist_easy/dcgan',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 5,
        """number of epochs to run""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

# leaky relu
def lrelu(x, leak=0.2):
    f1 = 0.5*(1 + leak)
    f2 = 0.5*(1 - leak)
    return tf.nn.relu(x)

"""
# concat labels
def conv_concat(x, y):
    y_dim = y.get_shape()[1]
    y = tf.reshape(y, tf.pack([-1, 1, 1, y_dim]))
    x_shape = tf.shape(x)
    return tf.concat(3, [x,
        y*tf.ones(tf.pack([x_shape[0], x_shape[1], x_shape[2], y_dim]))])
"""

is_training = tf.placeholder(tf.bool)
def discriminate(x, reuse=None):
    with tf.variable_scope('Dis', reuse=reuse):
        out = conv(x, 32, [5, 5], stride=[2, 2], activation_fn=lrelu)
        out = conv_bn(out, 64, [5, 5], is_training, stride=[2, 2], activation_fn=lrelu)
        out = conv_bn(out, 128, [5, 5], is_training, stride=[2, 2], activation_fn=lrelu)
        out = fc_bn(out, 512, is_training, activation_fn=lrelu)
        out = linear(out, 1)
        return out

def generate(z):
    with tf.variable_scope('Gen'):
        out = fc_bn(z, 512, is_training)
        out = fc_bn(out, 4*4*128, is_training)
        out = tf.reshape(out, [-1, 4, 4, 128])
        out = deconv_bn(out, 64, [5, 5], is_training, stride=[2, 2])
        out = deconv_bn(out, 32, [5, 5], is_training, stride=[2, 2])
        out = deconv(out, 1, [5, 5], stride=[2, 2], activation_fn=tf.nn.sigmoid)
        return out

x = tf.placeholder(tf.float32, [None, 32*32])
z = tf.placeholder(tf.float32, [None, 100])

real = tf.reshape(x, [-1, 32, 32, 1])
logits_real = discriminate(real)
fake = generate(z)
logits_fake = discriminate(fake, reuse=True)

def bce(logits, zero_or_one):
    ref = tf.zeros_like(logits) if zero_or_one is 0 else tf.ones_like(logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, ref)
    loss = tf.reduce_mean(loss)
    return loss

L_D = bce(logits_real, 1) + bce(logits_fake, 0)
L_G = bce(logits_fake, 1)

# get train ops
learning_rate = tf.placeholder(tf.float32)
vars = tf.trainable_variables()
D_vars = [var for var in vars if 'Dis' in var.name]
G_vars = [var for var in vars if 'Gen' in var.name]

train_D = get_train_op(L_D, var_list=D_vars, grad_clip=10.,
        learning_rate=learning_rate, beta1=0.5)

train_G = get_train_op(L_G, var_list=G_vars, grad_clip=10.,
        learning_rate=learning_rate, beta1=0.5)

# load data
from utils.data import load_pkl
train_xy, test_xy, _ = load_pkl('data/dmnist/dmnist_easy.pkl.gz')
train_x, train_y = train_xy
test_x, test_y = test_xy
batch_size = 100
n_train_batches = len(train_x)/batch_size
n_test_batches = len(test_x)/batch_size

saver = tf.train.Saver()
sess = tf.Session()
idx = range(len(train_x))
def train():
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    sess.run(tf.initialize_all_variables())
    lr_D = 0.0001
    lr_G = 0.001
    start = time.time()
    for i in range(FLAGS.n_epochs):
        np.random.shuffle(idx)
        train_L_D = 0.
        train_L_G = 0.
        start = time.time()
        for j in range(n_train_batches):
            batch_x = train_x[idx[j*batch_size:(j+1)*batch_size]]
            batch_z = np.random.uniform(-1, 1, [batch_size, 100]).astype(np.float32)

            # train discriminator
            _, batch_L_D = sess.run([train_D, L_D],
                    {x:batch_x, z:batch_z, learning_rate:lr_D, is_training:True})

            # train generator
            _, batch_L_G = sess.run([train_G, L_G],
                    {z:batch_z, learning_rate:lr_G, is_training:True})

            train_L_D += batch_L_D
            train_L_G += batch_L_G

            if (j+1)%50 == 0:
                accum_L_D = train_L_D / (j+1)
                accum_L_G = train_L_G / (j+1)
                line = 'Epoch %d Iter %d (%.3f secs), train L_D %f, train L_G %f' \
                        %(i+1, j+1, time.time()-start, accum_L_D, accum_L_G)
                print line
        train_L_D /= n_train_batches
        train_L_G /= n_train_batches

        test_L_D = 0.
        test_L_G = 0.
        for j in range(n_test_batches):
            batch_x = test_x[j*batch_size:(j+1)*batch_size]
            batch_z = np.random.uniform(-1, 1, [batch_size, 100]).astype(np.float32)

            test_L_D += sess.run(L_D, {x:batch_x, z:batch_z, is_training:False})
            test_L_G += sess.run(L_G, {z:batch_z, is_training:False})
        test_L_D /= n_test_batches
        test_L_G /= n_test_batches

        line = 'Epoch %d (%.3f secs), train L_D %f, train L_G %f, test L_D %f, test L_G %f\n' \
                %(i+1, time.time()-start, train_L_D, train_L_G, test_L_D, test_L_G)
        print line
        logfile.write(line + '\n')

    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')
    fig = plt.figure('generated')
    batch_z = np.random.uniform(-1, 1, [100, 100])
    gen = sess.run(fake, {z:batch_z, is_training:False})
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
