import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchimg_to_tileimg
from utils.misc import Logger
from utils.distribution import *
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
c_dim = 10
# noise latent variables
z_dim = 64

height = 28
width = 28

is_training = tf.placeholder(tf.bool)
def discriminate(x, reuse=None):
    with tf.variable_scope('Dis', reuse=reuse):
        out = conv(x, 64, 4, stride=2, activation_fn=lrelu)
        out = conv_bn(out, 128, 4, is_training, stride=2, activation_fn=lrelu)
        out = fc_bn(flat(out), 1024, is_training)
        # logits for discriminator probability
        logits = linear(out, 1)
        return logits, out

qc = Categorical(c_dim)
def recognize(out, reuse=None):
    with tf.variable_scope('Rec', reuse=reuse):
        out = fc_bn(out, 128, is_training, activation_fn=lrelu)
        out = linear(out, c_dim)
        return qc.get_param(out)

def generate(c, z, reuse=None):
    with tf.variable_scope('Gen', reuse=reuse):
        out = fc_bn(tf.concat(1, [c, z]), 1024, is_training)
        out = fc_bn(out, height/4*width/4*128, is_training)
        out = tf.reshape(out, [-1, height/4, width/4, 128])
        out = deconv_bn(out, 64, 4, is_training, stride=2)
        out = deconv(out, 1, 4, stride=2, activation_fn=tf.nn.sigmoid)
        return out

def bce(logits, zero_or_one):
    ref = tf.zeros_like(logits) if zero_or_one is 0 else tf.ones_like(logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, ref)
    loss = tf.reduce_mean(loss)
    return loss

def gen_code(batch_size):
    return qc.prior_sample(batch_size)

def gen_noise(batch_size, lim=1.):
    return np.random.uniform(-lim, lim, [batch_size, z_dim]).astype(np.float32)

x = tf.placeholder(tf.float32, [None, height*width])
c = tf.placeholder(tf.float32, [None, c_dim])
z = tf.placeholder(tf.float32, [None, z_dim])

real = tf.reshape(x, [-1, height, width, 1])

# discriminate real image
logits_real, _ = discriminate(real)

# generate fake images
fake = generate(c, z)
logits_fake, fake_out = discriminate(fake, reuse=True)
qc_param = recognize(fake_out)

# losses
lam = 1.
L_D = bce(logits_real, 1) + bce(logits_fake, 0)
L_G = bce(logits_fake, 1)
L_I = -qc.log_likel(c, qc_param)
                
# get train ops
learning_rate = tf.placeholder(tf.float32)
vars = tf.trainable_variables()
Dis_vars = [var for var in vars if 'Dis' in var.name]
Gen_vars = [var for var in vars if 'Gen' in var.name]
Rec_vars = [var for var in vars if 'Rec' in var.name]
train_D = get_train_op(L_D, var_list=Dis_vars, grad_clip=10.,
        learning_rate=learning_rate, beta1=0.5)
train_G = get_train_op(L_G+lam*L_I, var_list=Gen_vars+Rec_vars, grad_clip=10.,
        learning_rate=learning_rate, beta1=0.5)

# load data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_test_batches = mnist.test.num_examples / batch_size

saver = tf.train.Saver()
sess = tf.Session()
def train():
    train_Logger = Logger('train L_D', 'train L_G', 'train L_I')
    test_Logger = Logger('test L_D', 'test L_G', 'test L_I')
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    sess.run(tf.initialize_all_variables())
    lr_D = 0.0002
    lr_G = 0.001
    start = time.time()
    for i in range(FLAGS.n_epochs):
        train_Logger.clear()
        start = time.time()
        for j in range(n_train_batches):
            # train discriminator
            batch_x, _ = mnist.train.next_batch(batch_size)
            batch_c = gen_code(batch_size)
            batch_z = gen_noise(batch_size)
            feed_dict = {x:batch_x, c:batch_c, z:batch_z, 
                    learning_rate:lr_D, is_training:True}
            D_res = sess.run([train_D, L_D], feed_dict)

            # train generator
            batch_c = gen_code(batch_size)
            batch_z = gen_noise(batch_size)
            feed_dict = {x:batch_x, c:batch_c, z:batch_z, 
                    learning_rate:lr_G, is_training:True}
            G_res = sess.run([train_G, L_G, L_I], feed_dict)

            train_Logger.accum(D_res + G_res)

            if (j+1)%50 == 0:
                print train_Logger.get_status(i+1, time.time()-start, it=j+1)

        test_Logger.clear()
        for j in range(n_test_batches):
            batch_x, _ = mnist.test.next_batch(batch_size)
            batch_c = gen_code(batch_size)
            batch_z = gen_noise(batch_size)
            feed_dict = {x:batch_x, c:batch_c, 
                    z:batch_z, is_training:False}
            test_Logger.accum(sess.run([L_D, L_G, L_I], feed_dict))

        line = train_Logger.get_status(i+1, time.time()-start) + \
                test_Logger.get_status_no_header() + '\n'
        print line
        logfile.write(line + '\n')

    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')

    fig = plt.figure('generated')
    batch_c = np.repeat(np.eye(10), 10, axis=0)
    batch_z = gen_noise(100)
    feed_dict = {c:batch_c, z:batch_z, is_training:False}
    gen = sess.run(fake, feed_dict)
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
