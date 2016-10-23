import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from utils.nn import *
from utils.distribution import *
from utils.image import batchmat_to_tileimg, batchimg_to_tileimg
from utils.misc import Logger
#from stn.spatial_transformer import *
from attention.spatial_transformer import *
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', './results',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 10,
        """number of epochs to run""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

height = 56
width = 56
n_cls = 15
N = height/3
x = tf.placeholder(tf.float32, shape=[None, height*width])
x_img = tf.reshape(x, [-1, height, width, 1])
y = tf.placeholder(tf.int32, shape=[None])
y_one_hot = one_hot(y, n_cls)
is_training = tf.placeholder(tf.bool)


# localization net
rnn = tf.nn.rnn_cell.GRUCell(200)
state = rnn.zero_state(tf.shape(x)[0], tf.float32)

loc = pool(x_img, 2)
loc = conv(loc, 20, 3)
loc = pool(loc, 2)
loc = conv(loc, 20, 3)
loc = fc(flat(loc), 200)
with tf.variable_scope('rnn'):
    hid, state = rnn(flat(loc), state)
loc_param0 = to_loc(hid, s_max=0.4)
with tf.variable_scope('rnn', reuse=True):
    hid, state = rnn(flat(loc), state)
loc_param1 = to_loc(hid, s_max=0.4)

trans0 = spatial_transformer(x_img, loc_param0, height/2, width/2)
trans1 = spatial_transformer(x_img, loc_param1, height/2, width/2)

net0 = pool(conv_bn(trans0, 16, 3, is_training), 2)
net0 = pool(conv_bn(net0, 32, 3, is_training), 2)
net0 = pool(conv_bn(net0, 64, 3, is_training), 2)

net1 = pool(conv_bn(trans1, 16, 3, is_training), 2)
net1 = pool(conv_bn(net1, 32, 3, is_training), 2)
net1 = pool(conv_bn(net1, 64, 3, is_training), 2)

net = tf.concat(1, [flat(net0), flat(net1)])
net = fc_bn(net, 256, is_training)
net = linear(net, n_cls)

"""
rnn = tf.nn.rnn_cell.GRUCell(200)
state = rnn.zero_state(tf.shape(x)[0], tf.float32)
with tf.variable_scope('rnn'):
    hid, state = rnn(flat(loc), state)
loc_param0 = to_loc(hid)
trans0 = spatial_transformer(x_img, loc_param0, height/2, width/2)
with tf.variable_scope('rnn', reuse=True):
    hid, state = rnn(flat(loc), state)
loc_param1 = to_loc(hid)
trans1 = spatial_transformer(x_img, loc_param1, height/2, width/2)
"""

cent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, y_one_hot))
correct = tf.equal(tf.argmax(net, 1), tf.argmax(y_one_hot, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))
train_op = get_train_op(cent, grad_clip=10.)

from utils.data import load_pkl
train_xy, test_xy, _ = load_pkl('data/dmnist/dmnist.pkl.gz')
train_x, train_y = train_xy
test_x, test_y = test_xy
batch_size = 100
n_train_batches = len(train_x) / batch_size
n_test_batches = len(test_x) / batch_size

sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

def train():
    idx = range(len(train_x))
    train_Logger = Logger('train cent', 'train acc')
    test_Logger = Logger('test acc')
    logfile = open(FLAGS.save_dir + '/train.log', 'w', 0)
    sess.run(tf.initialize_all_variables())
    for i in range(FLAGS.n_epochs):
        np.random.shuffle(idx)
        train_Logger.clear()
        start = time.time()
        for j in range(n_train_batches):
            batch_idx = idx[j*batch_size:(j+1)*batch_size]
            batch_x = train_x[batch_idx]
            batch_y = train_y[batch_idx]
            feed_dict = {x:batch_x, y:batch_y, is_training:True}
            train_Logger.accum(sess.run([train_op, cent, acc], feed_dict))

        test_Logger.clear()
        for j in range(n_test_batches):
            batch_idx = range(j*batch_size, (j+1)*batch_size)
            batch_x = test_x[batch_idx]
            batch_y = test_y[batch_idx]
            feed_dict = {x:batch_x, y:batch_y, is_training:False}
            test_Logger.accum(sess.run(acc, feed_dict))

        line = train_Logger.get_status(i+1, time.time()-start) + \
                test_Logger.get_status_no_header()
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

    fig = plt.figure('first attention')
    plt.gray()
    plt.axis('off')
    att0 = sess.run(trans0, {x:batch_x, is_training:False})
    plt.imshow(batchimg_to_tileimg(att0, (10, 10)))
    fig.savefig(FLAGS.save_dir+'/att0.png')

    att1 = sess.run(trans1, {x:batch_x, is_training:False})
    fig = plt.figure('second attention')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchimg_to_tileimg(att1, (10, 10)))
    fig.savefig(FLAGS.save_dir+'/att1.png')

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
