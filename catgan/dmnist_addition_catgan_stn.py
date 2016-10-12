import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchimg_to_tileimg
import time
import os
import matplotlib.pyplot as plt
from stn.spatial_transformer import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/dmnist_addition/catgan_stn',
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

def localize(x, reuse=None):
    with tf.variable_scope('Loc', reuse=reuse):
        out = pool(x, [2, 2])
        out = conv(out, 5, [5, 5])
        out = pool(out, [2, 2])
        out = conv(out, 10, [5, 5])
        return to_loc(out)

# losses
def cross_entropy(p, y_one_hot, tol=1e-10):
    log_p = tf.log(p + tol)
    return -tf.reduce_mean(tf.reduce_sum(y_one_hot*log_p, 1))

def entropy(p, tol=1e-10):
    log_p = tf.log(p + tol)
    return -tf.reduce_mean(tf.reduce_sum(p*log_p, 1))

def discriminate(x_loc, reuse=None):
    with tf.variable_scope('Dis', reuse=reuse):
        out = conv(x, 32, [5, 5], activation_fn=lrelu)
        out = pool(out, [3, 3], [2, 2])
        out = conv_bn(out, 64, [3, 3], is_train, activation_fn=lrelu)
        out = conv_bn(out, 64, [3, 3], is_train, activation_fn=lrelu)




L_D = cross_entropy(p_real, y_one_hot) - entropy(p_fake)
L_G = -entropy(batch_p_fake) + entropy(p_fake)

# get train ops
learning_rate = tf.placeholder(tf.float32)
vars = tf.trainable_variables()
loc_vars = [var for var in vars if 'Loc' in var.name]
D_vars = [var for var in vars if 'Dis' in var.name]
G_vars = [var for var in vars if 'Gen' in var.name]

train_D = get_train_op(L_D, var_list=D_vars+loc_vars,
        learning_rate=learning_rate, grad_clip=10.)

train_G = get_train_op(L_G, var_list=G_vars,
        learning_rate=learning_rate, grad_clip=10.)

# to check classification accuracy
correct = tf.equal(tf.argmax(p_real, 1), tf.argmax(y_one_hot, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

# load data
from utils.data import load_pkl
train_xy, valid_xy, test_xy = load_pkl('data/dmnist/dmnist.pkl.gz')
train_x, train_y = train_xy
valid_x, valid_y = valid_xy
test_x, test_y = test_xy
batch_size = 100
n_train_batches = len(train_x) / batch_size
n_valid_batches = len(valid_x) / batch_size

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
            batch_x = train_x[j*batch_size:(j+1)*batch_size]
            batch_y = train_y[j*batch_size:(j+1)*batch_size]

            # train discriminator
            batch_z = np.random.normal(size=(batch_size, 128))
            _, batch_L_D, batch_acc = sess.run([train_D, L_D, acc],
                    {x:batch_x, y:batch_y, z:batch_z, learning_rate:lr_D, is_train:True})

            # train generator
            batch_z = np.random.normal(size=(batch_size, 128))
            _, batch_L_G = sess.run([train_G, L_G],
                    {z:batch_z, learning_rate:lr_G, is_train:True})

            train_L_D += batch_L_D
            train_acc += batch_acc
            train_L_G += batch_L_G

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
    fig = plt.figure('localized')
    loc = sess.run(x_loc, {x:test_x[0:100], is_train:False})
    plt.gray()
    plt.axis('off')
    plt.imshow(batchimg_to_tileimg(loc, (10, 10)))
    fig.savefig(FLAGS.save_dir+'/localized.png')

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
