import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchimg_to_tileimg
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/infogan',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 5,
        """number of epochs to run""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

# code spec
c0_dim = 10
c0_range = 1.
c0_range = 1.

# noise latent variables
z_dim = 62

# leaky relu
def lrelu(x, leak=0.1):
    f1 = 0.5*(1 + leak)
    f2 = 0.5*(1 - leak)
    return tf.nn.relu(x)

is_training = tf.placeholder(tf.bool)
def discriminate(x, reuse=None):
    with tf.variable_scope('Dis', reuse=reuse):
        out = conv(x, 64, [4, 4], stride=[2, 2], activation_fn=lrelu)
        out = conv_bn(out, 128, [4, 4], is_training, stride=[2, 2], activation_fn=lrelu)
        out = fc_bn(flat(out), 1024, is_training)
        # logits for discriminator probability
        logits = linear(out, 1)
        return logits, out

def recognize(out, reuse=None):
    with tf.variable_scope('Rec', reuse=reuse):
        out = fc_bn(out, 128, is_training, activation_fn=lrelu)
        out = linear(out, c0_dim+2)
        c0_param = tf.nn.softmax(tf.slice(out, [0,0], [-1,c0_dim]))
        c1_param = tf.slice(out, [0,c0_dim], [-1,1])
        c2_param = tf.slice(out, [0,c0_dim+1], [-1,1])
        return c0_param, c1_param, c2_param

def generate(c0, c1, c2, z, reuse=None):
    with tf.variable_scope('Gen', reuse=reuse):
        out = fc_bn(tf.concat(1, [c0, c1, c2, z]), 1024, is_training)
        out = fc_bn(out, 7*7*128, is_training)
        out = tf.reshape(out, [-1, 7, 7, 128])
        out = deconv_bn(out, 64, [4, 4], is_training, stride=[2, 2])
        out = deconv(out, 1, [4, 4], stride=[2, 2], activation_fn=tf.nn.sigmoid)
        return out

def bce(logits, zero_or_one):
    ref = tf.zeros_like(logits) if zero_or_one is 0 else tf.ones_like(logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, ref)
    loss = tf.reduce_mean(loss)
    return loss

def gaussian_ll(x, mean, log_var=None):
    c = -0.5*np.log(2*np.pi)
    log_var = tf.zeros_like(mean) if log_var is None else log_var
    var = tf.exp(log_var)
    ll = tf.reduce_sum(c-0.5*log_var-0.5*tf.square(x-mean)/var, 1)
    return tf.reduce_mean(ll)

def cat_ll(x, p):
    ll = tf.reduce_sum(tf.log(p + 1e-10)*x, 1)
    return tf.reduce_mean(ll)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
c0_int = tf.placeholder(tf.int32, [None])
c0 = one_hot(c0_int, c0_dim)
c1 = tf.placeholder(tf.float32, [None, 1])
c2 = tf.placeholder(tf.float32, [None, 1])
z = tf.placeholder(tf.float32, [None, z_dim])

real = tf.reshape(x, [-1,28,28,1])
# discriminate real image
logits_real, _ = discriminate(real)

# generate fake images
fake = generate(c0, c1, c2, z)
logits_fake, fake_out = discriminate(fake, reuse=True)
c0_param, c1_param, c2_param = recognize(fake_out)


"""
# get mutual information regularization
c0_param, c1_param, c2_param = recognize(fake, reuse_shared=True)
"""

# losses
lam_cat = 1.
lam_unif = 1.
L_D = bce(logits_real, 1) + bce(logits_fake, 0)
L_G = bce(logits_fake, 1)
L_I = -lam_cat*cat_ll(c0, c0_param) \
        -lam_unif*gaussian_ll(c1, c1_param) \
        -lam_unif*gaussian_ll(c2, c2_param)
                
# get train ops
learning_rate = tf.placeholder(tf.float32)
vars = tf.trainable_variables()
Dis_vars = [var for var in vars if 'Dis' in var.name]
Gen_vars = [var for var in vars if 'Gen' in var.name]
Rec_vars = [var for var in vars if 'Rec' in var.name]
train_D = get_train_op(L_D, var_list=Dis_vars, grad_clip=10.,
        learning_rate=learning_rate, beta1=0.5)
train_G = get_train_op(L_G+L_I, var_list=Gen_vars+Rec_vars, grad_clip=10.,
        learning_rate=learning_rate, beta1=0.5)

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
    lr_D = 0.0002
    lr_G = 0.001
    start = time.time()
    for i in range(FLAGS.n_epochs):
        start = time.time()
        train_L_D = 0.
        train_L_G = 0.        
        train_L_I = 0.
        for j in range(n_train_batches):
            batch_x, _ = mnist.train.next_batch(batch_size)

            # train discriminator
            batch_c0_int = np.random.randint(0, c0_dim, batch_size).astype(np.int32)
            batch_c1 = np.random.uniform(-1, 1, [batch_size, 1]).astype(np.float32)
            batch_c2 = np.random.uniform(-1, 1, [batch_size, 1]).astype(np.float32)
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            feed_dict = {x:batch_x, c0_int:batch_c0_int, c1:batch_c1, c2:batch_c2, z:batch_z, 
                    learning_rate:lr_D, is_training:True}
            _, batch_L_D = sess.run([train_D, L_D], feed_dict)

            # train generator
            batch_c0_int = np.random.randint(0, c0_dim, batch_size).astype(np.int32)
            batch_c1 = np.random.uniform(-1, 1, [batch_size, 1]).astype(np.float32)
            batch_c2 = np.random.uniform(-1, 1, [batch_size, 1]).astype(np.float32)
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            feed_dict = {x:batch_x, c0_int:batch_c0_int, c1:batch_c1, c2:batch_c2, z:batch_z,
                    learning_rate:lr_G, is_training:True}
            _, batch_L_G, batch_L_I = sess.run([train_G, L_G, L_I], feed_dict)

            train_L_D += batch_L_D
            train_L_G += batch_L_G
            train_L_I += batch_L_I

            if (j+1)%50 == 0:
                accum_L_D = train_L_D / (j+1)
                accum_L_G = train_L_G / (j+1)
                accum_L_I = train_L_I / (j+1)
                line = 'Epoch %d Iter %d, train L_D %f, train L_G %f, train L_I %f' \
                        %(i+1, j+1, accum_L_D, accum_L_G, accum_L_I)
                print line

        train_L_D /= n_train_batches
        train_L_G /= n_train_batches
        train_L_I /= n_train_batches
        line = 'Epoch %d (%.3f secs), train L_D %f, train L_G %f, train L_I %f\n' \
                %(i+1, time.time()-start, train_L_D, train_L_G, train_L_I)
        print line
        logfile.write(line + '\n')

    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')

    fig = plt.figure('varying c1')
    batch_c0_int = np.repeat(range(10), 10)
    batch_c1 = np.expand_dims(np.tile(np.linspace(-3, 3, 10), 10), 1)
    batch_c2 = np.expand_dims(np.tile(np.random.uniform(-1, 1), 100), 1)
    batch_z = np.tile(np.random.uniform(-1, 1, [1, z_dim]), [100, 1])
    feed_dict = {c0_int:batch_c0_int, c1:batch_c1, c2:batch_c2, z:batch_z, is_training:False}
    gen = sess.run(fake, feed_dict)
    plt.gray()
    plt.axis('off')
    plt.imshow(batchimg_to_tileimg(gen, (10, 10)))
    fig.savefig(FLAGS.save_dir+'/genereated_c1.png')

    fig = plt.figure('varying c2')
    batch_c0_int = np.repeat(range(10), 10)
    batch_c1 = np.expand_dims(np.tile(np.random.uniform(-1, 1), 100), 1)
    batch_c2 = np.expand_dims(np.tile(np.linspace(-3, 3, 10), 10), 1)
    batch_z = np.tile(np.random.uniform(-1, 1, [1, z_dim]), [100, 1])
    feed_dict = {c0_int:batch_c0_int, c1:batch_c1, c2:batch_c2, z:batch_z, is_training:False}
    gen = sess.run(fake, feed_dict)
    plt.gray()
    plt.axis('off')
    plt.imshow(batchimg_to_tileimg(gen, (10, 10)))
    fig.savefig(FLAGS.save_dir+'/genereated_c2.png')

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
