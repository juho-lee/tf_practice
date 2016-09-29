import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
from utils.image import batchmat_to_tileimg
from utils.data import load_pkl
from draw.attention import *
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/tmnist/cvae_attn',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 20,
        """number of epochs to run""")
tf.app.flags.DEFINE_integer('ksize', 3,
        """convolution kernel size in encoder""")
tf.app.flags.DEFINE_integer('n_ch', 16,
        """number of hidden channels to start""")
tf.app.flags.DEFINE_integer('n_lat', 10,
        """number of latent variables""")
tf.app.flags.DEFINE_integer('N', 30,
        """attention size""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

ksize = FLAGS.ksize
n_ch = FLAGS.n_ch
n_lat = FLAGS.n_lat
N = FLAGS.N
height = 50
width = 50
attunit = AttentionUnit(height, width, 1, N)
n_in = height*width
x = tf.placeholder(tf.float32, shape=[None, n_in])
x_img = tf.reshape(x, [-1, height, width, 1])
hid_t_enc = conv(x_img, n_ch, [ksize, ksize], [2, 2])
hid_t_enc = conv(hid_t_enc, n_ch*2, [ksize, ksize], [2, 2])
hid_t_enc = conv(hid_t_enc, n_ch*4, [ksize, ksize], [2, 2])
hid_t_enc = conv(hid_t_enc, n_ch*8, [ksize, ksize], [2, 2], padding='VALID')
hid_t_enc = flat(hid_t_enc)
z_t_mean = linear(hid_t_enc, n_lat)
z_t_log_var = linear(hid_t_enc, n_lat)
z_t = gaussian_sample(z_t_mean, z_t_log_var)
trans = to_att(fc(z_t, 10))

x_att = attunit.read(x, trans)
x_att_img = tf.reshape(x_att, [-1, N, N, 1])
hid_c_enc = conv(x_att_img, n_ch, [ksize, ksize], [2, 2])
hid_c_enc = conv(hid_c_enc, n_ch*2, [ksize, ksize], [2, 2], padding='VALID')
hid_c_enc = conv(hid_c_enc, n_ch*4, [ksize, ksize], [2, 2], padding='VALID')
hid_c_enc = flat(hid_c_enc)
z_c_mean = linear(hid_c_enc, n_lat)
z_c_log_var = linear(hid_c_enc, n_lat)
z_c = gaussian_sample(z_c_mean, z_c_log_var)

hid_dec = fc(z_c, n_ch*4*3*3)
hid_dec = tf.reshape(hid_dec, [-1, 3, 3, n_ch*4])
hid_dec = deconv(hid_dec, n_ch*2, [3, 3], [2, 2], padding='VALID')
hid_dec = deconv(hid_dec, n_ch, [3, 3], [2, 2], padding='VALID')
p_att = flat(deconv(hid_dec, 1, [2, 2], [2, 2], activation_fn=tf.nn.sigmoid))
p = tf.clip_by_value(attunit.write(p_att, trans), 0, 1)

train_xy, valid_xy, test_xy = load_pkl('data/tmnist/tmnist.pkl.gz')
train_x, _ = train_xy
valid_x, _ = valid_xy
test_x, _ = test_xy
batch_size = 100
n_train_batches = len(train_x) / batch_size
n_valid_batches = len(valid_x) / batch_size

neg_ll = bernoulli_neg_ll(x, p)
kld = gaussian_kld(z_t_mean, z_t_log_var) + gaussian_kld(z_c_mean, z_c_log_var)
loss = neg_ll + kld

learning_rate = tf.placeholder(tf.float32)
train_op = get_train_op(loss, learning_rate=learning_rate, grad_clip=10.)
saver = tf.train.Saver()
sess = tf.Session()

def train():
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    logfile.write(('n_in: %d, n_ch: %d, n_lat: %d\n' % (n_in, n_ch, n_lat)))
    sess.run(tf.initialize_all_variables())
    lr = 0.001
    for i in range(FLAGS.n_epochs):
        start = time.time()
        train_neg_ll = 0.
        train_kld = 0.
        for j in range(n_train_batches):
            batch_x = train_x[j*batch_size:(j+1)*batch_size]
            _, batch_neg_ll, batch_kld = \
                    sess.run([train_op, neg_ll, kld], {x:batch_x, learning_rate:lr})
            train_neg_ll += batch_neg_ll
            train_kld += batch_kld
        train_neg_ll /= n_train_batches
        train_kld /= n_train_batches

        if (i+1) % 3 == 0:
            lr = lr * 0.8

        valid_neg_ll = 0.
        valid_kld = 0.
        for j in range(n_valid_batches):
            batch_x = valid_x[j*batch_size:(j+1)*batch_size]
            batch_neg_ll, batch_kld = sess.run([neg_ll, kld], {x:batch_x})
            valid_neg_ll += batch_neg_ll
            valid_kld += batch_kld
        valid_neg_ll /= n_valid_batches
        valid_kld /= n_valid_batches

        line = "Epoch %d (%f sec), train loss %f = %f + %f, valid loss %f = %f + %f" \
                % (i+1, time.time()-start,
                        train_neg_ll+train_kld, train_neg_ll, train_kld,
                        valid_neg_ll+valid_kld, valid_neg_ll, valid_kld)
        print line
        logfile.write(line + '\n')
    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')
    batch_x = test_x[0:100]
    fig = plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(batch_x, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/original.png')

    batch_x_att, p_recon_att, p_recon = \
            sess.run([tf.clip_by_value(x_att, 0, 1), p_att, p], {x:batch_x})
    fig = plt.figure('attended')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(batch_x_att, (N, N), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/attended.png')
    fig = plt.figure('reconstructed_attended')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(p_recon_att, (N, N), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/reconstructed_attended.png')
    fig = plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(p_recon, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/reconstructed.png')

    t_noise = np.random.normal(size=(100, n_lat))
    c_noise = np.repeat(np.random.normal(size=(10, n_lat)), 10, axis=0)
    p_gen_att, p_gen = sess.run([p_att, p], {z_t:t_noise, z_c:c_noise})
    fig = plt.figure('generated_attended.png')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(p_gen_att, (N, N), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/generated_attended.png')
    fig = plt.figure('generated')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(p_gen, (height, width), (10, 10)))
    fig.savefig(FLAGS.save_dir+'/generated.png')

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
