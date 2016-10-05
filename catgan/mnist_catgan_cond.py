import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchimg_to_tileimg
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/catgan_cond',
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

is_train = tf.placeholder(tf.bool)
def discriminate(x, reuse=None):
    with tf.variable_scope('Dis', reuse=reuse):
        p = conv(x, 32, [5, 5], activation_fn=lrelu)
        p = pool(p, [3, 3], [2, 2])
        p = conv_bn(p, 64, [3, 3], is_train, activation_fn=lrelu)
        p = conv_bn(p, 64, [3, 3], is_train, activation_fn=lrelu)
        p = pool(p, [3, 3], [2, 2])
        p = conv_bn(p, 128, [3, 3], is_train, activation_fn=lrelu)
        p = conv_bn(p, 10, [1, 1], is_train, activation_fn=lrelu)
        p = fc(flat(p), 10, activation_fn=tf.nn.softmax)
        return p

# concat labels
def conv_concat(x, y):
    y_dim = y.get_shape()[1]
    y = tf.reshape(y, tf.pack([-1, 1, 1, y_dim]))
    x_shape = tf.shape(x)
    return tf.concat(3, [x,
        y*tf.ones(tf.pack([x_shape[0], x_shape[1], x_shape[2], y_dim]))])

def generate(z, y, reuse=None):
    with tf.variable_scope('Gen', reuse=reuse):
        z = tf.concat(1, [z, y])
        gen = fc(z, 4*4*128, activation_fn=lrelu)
        gen = tf.reshape(gen, [-1, 4, 4, 128])
        gen = conv_concat(gen, y)
        gen = deconv_bn(gen, 96, [3, 3], is_train, stride=[2, 2], activation_fn=lrelu)
        gen = conv_concat(gen, y)
        gen = deconv_bn(gen, 64, [5, 5], is_train, stride=[2, 2], activation_fn=lrelu)
        gen = conv_concat(gen, y)
        gen = deconv_bn(gen, 64, [5, 5], is_train, stride=[2, 2], activation_fn=lrelu)
        gen = conv_concat(gen, y)
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
fake = generate(z, y)
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

#L_D = cross_entropy(p_real, y) - entropy(p_fake)
#L_G = -entropy(batch_p_fake) + entropy(p_fake)

L_D = cross_entropy(p_real, y) - entropy(p_fake)
L_G = entropy(p_fake) - entropy(batch_p_fake) + 0.05*cross_entropy(p_fake, y)

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
correct = tf.equal(tf.argmax(p_real, 1), tf.argmax(y, 1))
acc_D = tf.reduce_mean(tf.cast(correct, tf.float32))
correct = tf.equal(tf.argmax(p_fake, 1), tf.argmax(y, 1))
acc_G = tf.reduce_mean(tf.cast(correct, tf.float32))

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
        train_acc_D = 0.
        train_L_G = 0.
        train_acc_G = 0.
        for j in range(n_train_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # train discriminator
            batch_z = np.random.normal(size=(batch_size, 128))
            _, batch_L_D, batch_acc_D = sess.run([train_D, L_D, acc_D],
                    {x:batch_x, y:batch_y, z:batch_z, learning_rate:lr_D, is_train:True})

            # train generator
            batch_z = np.random.normal(size=(batch_size, 128))
            _, batch_L_G, batch_acc_G = sess.run([train_G, L_G, acc_G],
                    {y:batch_y, z:batch_z, learning_rate:lr_G, is_train:True})

            train_L_D += batch_L_D
            train_acc_D += batch_acc_D
            train_L_G += batch_L_G
            train_acc_G += batch_acc_G

            if (j+1)%50 == 0:
                accum_L_D = train_L_D / (j+1)
                accum_acc_D = train_acc_D / (j+1)
                accum_L_G = train_L_G / (j+1)
                accum_acc_G = train_acc_G / (j+1)
                line = 'Epoch %d Iter %d, train L_D %f, train acc_D %f, train L_G %f, train acc_G %f' \
                        %(i+1, j+1, accum_L_D, accum_acc_D, accum_L_G, accum_acc_G)
                print line

        train_L_D /= n_train_batches
        train_acc_D /= n_train_batches
        train_L_G /= n_train_batches
        train_acc_G /= n_train_batches

        line = 'Epoch %d (%.3f secs), train L_D %f, train acc_D %f, train L_G %f, train acc_G %f\n' \
                %(i+1, time.time()-start, train_L_D, train_acc_D, train_L_G, train_acc_G)
        print line
        logfile.write(line + '\n')

    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')
    fig = plt.figure('generated')
    batch_z = np.random.normal(size=(100, 128))
    batch_y = np.zeros((100, 10))
    for i in range(10):
        batch_y[10*i:10*(i+1), i] = i
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
