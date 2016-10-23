import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchimg_to_tileimg, batchmat_to_tileimg
from utils.misc import Logger
from utils.distribution import *
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/mnist/ali',
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

# leaky relu
def lrelu(x, leak=0.1):
    f1 = 0.5*(1 + leak)
    f2 = 0.5*(1 - leak)
    return tf.nn.relu(x)

is_training = tf.placeholder(tf.bool)
#pc = MaxOutGaussian(fix_var=True)
pc = Gaussian(fix_var=True)
def gen_code(batch_size):
    return pc.prior_sample([batch_size, c_dim])

pz = Gaussian(fix_var=True)
def gen_noise(batch_size):
    return pz.prior_sample([batch_size, z_dim])

#encoder
def encode(x_img, reuse=None):
    with tf.variable_scope('Enc', reuse=reuse):
        out = conv_bn(x_img, 64, 4, is_training, stride=2, activation_fn=lrelu)
        out = conv_bn(out, 128, 4, is_training, stride=2, activation_fn=lrelu)
        out = fc_bn(flat(out), 1024, is_training)
        c_out = linear(out, c_dim)
        z_out = linear(out, z_dim)
        return pc.get_param(c_out), pz.get_param(z_out)

def decode(c, z, reuse=None):
    with tf.variable_scope('Dec', reuse=reuse):
        out = fc_bn(tf.concat(1, [c, z]), 1024, is_training)
        out = fc_bn(out, height/4*width/4*128, is_training)
        out = tf.reshape(out, [-1, height/4, width/4, 128])
        out = deconv_bn(out, 64, 4, is_training, stride=2)
        out = deconv(out, 1, 4, stride=2, activation_fn=tf.nn.sigmoid)
        return out

def discriminate(x_img, c, z, reuse=None):
    with tf.variable_scope('Dis', reuse=reuse):
        # D_x       
        D_x_out = conv_bn(x_img, 64, 4, is_training, stride=2, activation_fn=lrelu)
        D_x_out = conv_bn(D_x_out, 128, 4, is_training, stride=2, activation_fn=lrelu)
        D_x_out = fc_bn(flat(D_x_out), 1024, is_training)

        # D_z
        D_z_out = fc(tf.concat(1, [c, z]), 1024)
        D_z_out = fc(D_z_out, 512)

        out = fc(tf.concat(1, [D_x_out, D_z_out]), 512)
        out = linear(out, 1)
        return out

def bce(logits, zero_or_one):
    ref = tf.zeros_like(logits) if zero_or_one is 0 else tf.ones_like(logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, ref)
    loss = tf.reduce_mean(loss)
    return loss

x = tf.placeholder(tf.float32, [None, height*width])
c = tf.placeholder(tf.float32, [None, c_dim])
z = tf.placeholder(tf.float32, [None, z_dim])
x_img = tf.reshape(x, [-1, height, width, 1])

# encode
c_hat_param, z_hat_param = encode(x_img)
c_hat = pc.sample(c_hat_param)
z_hat = pz.sample(z_hat_param)

# decode
x_img_hat = decode(c_hat, z_hat)

# discriminate
rho_q = discriminate(x_img, c_hat, z_hat)
rho_p = discriminate(x_img_hat, c, z, reuse=True)

# losses
L_D = bce(rho_q, 1) + bce(rho_p, 0)
L_G = bce(rho_q, 0) + bce(rho_p, 1)
                
# get train ops
learning_rate = tf.placeholder(tf.float32)
vars = tf.trainable_variables()
Enc_vars = [var for var in vars if 'Enc' in var.name]
Dec_vars = [var for var in vars if 'Dec' in var.name]
Dis_vars = [var for var in vars if 'Dis' in var.name]

train_D = get_train_op(L_D, var_list=Dis_vars, grad_clip=10.,
        learning_rate=learning_rate, beta1=0.5)
train_G = get_train_op(L_G, var_list=Enc_vars+Dec_vars, grad_clip=10.,
        learning_rate=learning_rate, beta1=0.5)


# load data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_test_batches = mnist.test.num_examples / batch_size

saver = tf.train.Saver()
sess = tf.Session()
def train():
    train_Logger = Logger('train L_D', 'train L_G')
    test_Logger = Logger('test L_D', 'test L_G')
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    sess.run(tf.initialize_all_variables())
    lr_D = 0.0001
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
            G_res = sess.run([train_G, L_G], feed_dict)

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
            test_Logger.accum(sess.run([L_D, L_G], feed_dict))

        line = train_Logger.get_status(i+1, time.time()-start) + \
                test_Logger.get_status_no_header() + '\n'
        print line
        logfile.write(line + '\n')

        if (i+1)%3 == 0:
            lr_D *= 0.8

    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')

    batch_x, _ = mnist.test.next_batch(100)

    fig = plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(batch_x, (height, width), (10, 10)))

    recon = sess.run(x_img_hat, {x:batch_x, is_training:False})
    fig = plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchimg_to_tileimg(recon, (10, 10)))

    batch_c = np.repeat(np.eye(10), 10, axis=0)
    gen = sess.run(x_img_hat, {c_hat:batch_c, z_hat:gen_noise(100), is_training:False})
    plt.figure('generated')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchimg_to_tileimg(gen, (10, 10)))
    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
