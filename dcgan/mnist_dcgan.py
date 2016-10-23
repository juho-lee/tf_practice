import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchimg_to_tileimg
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/dcgan',
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

# concat labels
def conv_concat(x, y):
    y_dim = y.get_shape()[1]
    y = tf.reshape(y, tf.pack([-1, 1, 1, y_dim]))
    x_shape = tf.shape(x)
    return tf.concat(3, [x,
        y*tf.ones(tf.pack([x_shape[0], x_shape[1], x_shape[2], y_dim]))])

is_train = tf.placeholder(tf.bool)
def discriminate(x, y, reuse=None):
    with tf.variable_scope('Dis', reuse=reuse):
        out = conv_concat(x, y)
        out = conv(out, 64, [5, 5], stride=[2, 2], activation_fn=lrelu)
        out = conv_concat(out, y)
        out = conv_bn(out, 128, [5, 5], is_train,
                stride=[2, 2], activation_fn=lrelu)
        out = tf.concat(1, [flat(out), y])
        out = fc_bn(out, 1024, is_train, activation_fn=lrelu)
        out = tf.concat(1, [out, y])
        out = linear(out, 1)
        return out

def generate(z, y):
    with tf.variable_scope('Gen'):
        out = tf.concat(1, [z, y])
        out = fc_bn(out, 1024, is_train)
        out = tf.concat(1, [out, y])
        out = fc_bn(out, 128*7*7, is_train)
        out = tf.reshape(out, [-1, 7, 7, 128])
        out = conv_concat(out, y)
        out = deconv_bn(out, 64, [5, 5], is_train, stride=[2, 2])
        out = conv_concat(out, y)
        out = deconv(out, 1, [5, 5], stride=[2, 2], activation_fn=tf.nn.sigmoid)
        return out

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
z = tf.placeholder(tf.float32, [None, 100])

real = tf.reshape(x, [-1, 28, 28, 1])
logits_real = discriminate(real, y)
fake = generate(z, y)
logits_fake = discriminate(fake, y, reuse=True)

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
        learning_rate=learning_rate)

train_G = get_train_op(L_G, var_list=G_vars, grad_clip=10.,
        learning_rate=learning_rate)

# load data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_valid_batches = mnist.validation.num_examples / batch_size

saver = tf.train.Saver()
sess = tf.Session()
def train():
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    sess.run(tf.initialize_all_variables())
    lr_D = 0.0001
    lr_G = 0.001
    start = time.time()
    for i in range(FLAGS.n_epochs):
        start = time.time()
        train_L_D = 0.
        train_L_G = 0.
        for j in range(n_train_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_z = np.random.uniform(-1, 1, [batch_size, 100]).astype(np.float32)

            # train discriminator
            _, batch_L_D = sess.run([train_D, L_D],
                    {x:batch_x, y:batch_y, z:batch_z, learning_rate:lr_D, is_train:True})

            # train generator
            _, batch_L_G = sess.run([train_G, L_G],
                    {z:batch_z, y:batch_y, learning_rate:lr_G, is_train:True})

            train_L_D += batch_L_D
            train_L_G += batch_L_G

            if (j+1)%50 == 0:
                accum_L_D = train_L_D / (j+1)
                accum_L_G = train_L_G / (j+1)
                line = 'Epoch %d Iter %d, train L_D %f, train L_G %f' \
                        %(i+1, j+1, accum_L_D, accum_L_G)
                print line
        train_L_D /= n_train_batches
        train_L_G /= n_train_batches

        valid_L_D = 0.
        valid_L_G = 0.
        for j in range(n_valid_batches):
            batch_x, batch_y = mnist.validation.next_batch(batch_size)
            batch_z = np.random.uniform(-1, 1, [batch_size, 100]).astype(np.float32)

            valid_L_D += sess.run(L_D, {x:batch_x, y:batch_y, z:batch_z, is_train:False})
            valid_L_G += sess.run(L_G, {z:batch_z, y:batch_y, is_train:False})
        valid_L_D /= n_valid_batches
        valid_L_G /= n_valid_batches

        line = 'Epoch %d (%.3f secs), train L_D %f, train L_G %f, valid L_D %f, valid L_G %f\n' \
                %(i+1, time.time()-start, train_L_D, train_L_G, valid_L_D, valid_L_G)
        print line
        logfile.write(line + '\n')

    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')
    fig = plt.figure('generated')
    batch_z = np.random.uniform(-1, 1, [100, 100])
    batch_y = np.zeros((100, 10))
    for i in range(10):
        batch_y[i*10:(i+1)*10,i] = i
    gen = sess.run(fake, {z:batch_z, y:batch_y, is_train:False})
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
