import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchimg_to_tileimg
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/dmnist/infogan',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 5,
        """number of epochs to run""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

c_dim = 10
z_dim = 70

# leaky relu
def lrelu(x, leak=0.1):
    f1 = 0.5*(1 + leak)
    f2 = 0.5*(1 - leak)
    return tf.nn.relu(x)

is_training = tf.placeholder(tf.bool)
def discriminate(x, reuse=None):
    with tf.variable_scope('Dis', reuse=reuse):
        out = conv(x, 32, [4, 4], stride=[2, 2], activation_fn=lrelu)
        out = conv_bn(out, 64, [4, 4], is_training, stride=[2, 2], activation_fn=lrelu)
        out = conv_bn(out, 128, [4, 4], is_training, stride=[2, 2], activation_fn=lrelu)
        out = fc_bn(flat(out), 1024, is_training)
        # logits for discriminator probability
        logits = linear(out, 1)
        return logits, out

def recognize(out, reuse=None):
    with tf.variable_scope('Rec', reuse=reuse):
        out = fc_bn(out, 128, is_training, activation_fn=lrelu)
        q_mean = linear(out, c_dim)
        return q_mean

def generate(c, z, reuse=None):
    with tf.variable_scope('Gen', reuse=reuse):
        out = fc_bn(tf.concat(1, [c, z]), 1024, is_training)
        out = fc_bn(out, 7*7*128, is_training)
        out = tf.reshape(out, [-1, 7, 7, 128])
        out = deconv_bn(out, 64, [4, 4], is_training, stride=[2, 2])
        out = deconv_bn(out, 32, [4, 4], is_training, stride=[2, 2])
        out = deconv(out, 1, [4, 4], stride=[2, 2], activation_fn=tf.nn.sigmoid)
        return out

def bce(logits, zero_or_one):
    ref = tf.zeros_like(logits) if zero_or_one is 0 else tf.ones_like(logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, ref)
    loss = tf.reduce_mean(loss)
    return loss

def rect_gaussian_ll(x, mean, log_var=None):
    def phi(x):
        return tf.exp(-0.5*tf.square(x))/np.sqrt(2*np.pi)
    def Phi(x):
        return 0.5 + 0.5*tf.erf(x/np.sqrt(2))

    log_var = tf.zeros_like(mean) if log_var is None else log_var
    var = tf.exp(log_var)
    log_std = 0.5*log_var
    std = tf.exp(log_std)
    
    p_zero = Phi(-mean/std)
    p_nonzero = phi((x-mean)/std)/std
    ll = tf.select(tf.equal(x, 0), p_zero, p_nonzero)
    ll = tf.reduce_sum(ll, 1)
    ll = tf.reduce_mean(ll)
    return ll

x = tf.placeholder(tf.float32, [None, 56*56])
c = tf.placeholder(tf.float32, [None, c_dim])
z = tf.placeholder(tf.float32, [None, z_dim])

real = tf.reshape(x, [-1,56,56,1])
# discriminate real image
logits_real, _ = discriminate(real)

# generate fake images
fake = generate(c, z)
logits_fake, fake_out = discriminate(fake, reuse=True)
q_mean = recognize(fake_out)

# losses
lam = 1.
L_D = bce(logits_real, 1) + bce(logits_fake, 0)
L_G = bce(logits_fake, 1)
L_I = -lam*rect_gaussian_ll(c, q_mean)
                
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
from utils.data import load_pkl
train_xy, valid_xy, _ = load_pkl('data/dmnist/dmnist_addition.pkl.gz')
train_x, train_y = train_xy
valid_x, valid_y = valid_xy
batch_size = 100
n_train_batches = len(train_x)/batch_size
n_valid_batches = len(valid_x)/batch_size

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
        train_L_I = 0.
        for j in range(n_train_batches):
            batch_x = train_x[j*batch_size:(j+1)*batch_size]

            # train discriminator
            batch_c = np.random.normal(size=(batch_size, c_dim))
            batch_c = 0.5*(batch_c + abs(batch_c))
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            feed_dict = {x:batch_x, c:batch_c, z:batch_z, learning_rate:lr_D, is_training:True}
            _, batch_L_D = sess.run([train_D, L_D], feed_dict)

            # train generator
            batch_c = np.random.normal(size=(batch_size, c_dim))
            batch_c = 0.5*(batch_c + abs(batch_c))
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            feed_dict = {x:batch_x, c:batch_c, z:batch_z, learning_rate:lr_G, is_training:True}
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

    fig = plt.figure('generated')
    #batch_c = np.random.normal(size=(100,c_dim))
    #batch_c = 0.5*(batch_c+abs(batch_c))
    batch_c = np.zeros((100, c_dim))
    for i in range(10):
        batch_c[i*10:(i+1)*10, i] = 1.
    batch_z = np.random.uniform(-1, 1, (100, z_dim))
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
