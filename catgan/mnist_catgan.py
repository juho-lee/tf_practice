import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchimg_to_tileimg
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/catgan',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 20,
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
#    return f1*x + f2*abs(x)

is_train = tf.placeholder(tf.bool)
def discriminate(x, reuse=None):
    with tf.variable_scope('Discriminator', reuse=reuse):
        p = conv_bn(x, 32, [5, 5], is_train, activation_fn=lrelu)
        p = pool(p, [3, 3], [2, 2])
        p = conv_bn(p, 64, [3, 3], is_train, activation_fn=lrelu)
        p = conv_bn(p, 64, [3, 3], is_train, activation_fn=lrelu)
        p = pool(p, [3, 3], [2, 2])
        p = conv_bn(p, 128, [3, 3], is_train, activation_fn=lrelu)
        p = conv_bn(p, 10, [1, 1], is_train, activation_fn=lrelu)
        p = dropout(p, keep_prob=0.5, is_training=is_train)
        p = fc(flat(p), 10, activation_fn=tf.nn.softmax)
        # p = fc_bn(flat(p), 10, is_train, activation_fn=tf.nn.softmax)
        return p

def generate(z, reuse=None):
    with tf.variable_scope('Generator', reuse=reuse):

        gen = fc_bn(z, 8*8*96, is_train, activation_fn=lrelu)
        gen = tf.reshape(gen, [-1, 8, 8, 96])
        """
        z = tf.reshape(z, [-1, 1, 1, 128])
        gen = deconv_bn(z, 96, [1, 1], is_train, stride=[4, 4], activation_fn=lrelu)
        gen = deconv_bn(gen, 96, [3, 3], is_train, stride=[2, 2], activation_fn=lrelu)
        """
        gen = deconv_bn(gen, 64, [5, 5], is_train, stride=[2, 2], activation_fn=lrelu)
        gen = deconv_bn(gen, 64, [5, 5], is_train, stride=[2, 2], activation_fn=lrelu)
        # gen = conv_bn(gen, 1, [5, 5], is_train, padding='VALID', activation_fn=tf.nn.sigmoid)
        gen = conv(gen, 1, [5, 5], padding='VALID', activation_fn=tf.nn.sigmoid)
        return gen

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
z = tf.placeholder(tf.float32, [None, 128])
x_img = tf.reshape(x, [-1,28,28,1])

# discriminator output for real image
p_real = discriminate(x_img)
batch_p_real = tf.reduce_mean(p_real, 0, keep_dims=True)
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

# L_D = entropy(p_real) - entropy(batch_p_real) - entropy(p_fake) \
#        + cross_entropy(p_real, y)
L_D = cross_entropy(p_real, y) - entropy(p_fake)
L_G = -entropy(batch_p_fake) + entropy(p_fake)

# get train ops
learning_rate = tf.placeholder(tf.float32)
vars = tf.trainable_variables()
D_params = [var for var in vars if 'Discriminator' in var.name]
G_params = [var for var in vars if 'Generator' in var.name]

"""
# weight decay???
weights_norm = tf.reduce_sum(0.0001*tf.pack(
    [tf.nn.l2_loss(w) for w in D_params]))
"""

train_D = get_train_op(L_D, var_list=D_params,
        learning_rate=learning_rate, grad_clip=10.)

train_G = get_train_op(L_G, var_list=G_params,
        learning_rate=learning_rate, grad_clip=10.)

# to check classification accuracy
correct = tf.equal(tf.argmax(p_real, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

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
        train_acc = 0.
        train_L_G = 0.
        for j in range(n_train_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # train discriminator
            batch_z = np.random.normal(size=(batch_size, 128))
            _, batch_L_D, batch_acc = sess.run([train_D, L_D, acc],
                    {x:batch_x, y:batch_y, z:batch_z, learning_rate:lr_D, is_train:True})

            # train generator
            for k in range(5):
                batch_z = np.random.normal(size=(batch_size, 128))
                _, batch_L_G = sess.run([train_G, L_G],
                        {z:batch_z, learning_rate:lr_G, is_train:True})

            train_L_D += batch_L_D
            train_acc += batch_acc
            train_L_G += batch_L_G

            if (j+1)%50 == 0:
                lr_D = lr_D * 0.8

            if (j+1)%50 == 0:
                accum_L_D = train_L_D / (j+1)
                accum_acc = train_acc / (j+1)
                accum_L_G = train_L_G / (j+1)
                line = 'Epoch %d Iter %d, train L_D %f, train acc %f, train L_G %f' \
                        %(i+1, j+1, accum_L_D, accum_acc, accum_L_G)
                print line

        train_L_D /= n_train_batches
        train_acc /= n_train_batches
        train_L_G /= n_train_batches
        line = 'Epoch %d (%.3f secs), train L_D %f, train acc %f, train L_G %f\n' \
                %(i+1, time.time()-start, train_L_D, train_acc, train_L_G)
        print line
        logfile.write(line + '\n')

    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')
    fig = plt.figure('generated')
    gen = sess.run(fake, {z:np.random.normal(size=(100, 128)), is_train:False})
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
