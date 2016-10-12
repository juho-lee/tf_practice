import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.prob import *
from utils.nn import *
import time
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', '../results/dmnist_addition/cnn',
        """directory to save models.""")
tf.app.flags.DEFINE_integer('n_epochs', 5,
        """number of epochs to run""")
tf.app.flags.DEFINE_boolean('train', True,
        """training (True) vs testing (False)""")

if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

x = tf.placeholder(tf.float32, [None, 56*56])
x_img = tf.reshape(x, [-1, 56, 56, 1])
y = tf.placeholder(tf.int32, [None])
y_one_hot = one_hot(y, 15)

# cnn
is_training = tf.placeholder(tf.bool)
net = pool(conv(x_img, 32, [3, 3]), [2, 2])
net = pool(conv_bn(net, 32, [3, 3], is_training), [2, 2])
net = pool(conv_bn(net, 64, [3, 3], is_training), [2, 2])
net = pool(conv_bn(net, 128, [3, 3], is_training), [2, 2])
net = fc_bn(flat(net), 1000, is_training)
code = fc_bn(net, 10, is_training)
net = linear(code, 15)

cent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, y_one_hot))
l1_loss = tf.reduce_mean(tf.reduce_sum(abs(code), 1))

train_op = get_train_op(cent + 0.1*l1_loss)
correct = tf.equal(tf.argmax(y_one_hot, 1), tf.argmax(net, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

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
    for i in range(FLAGS.n_epochs):
        train_cent = 0.
        train_acc = 0.
        start = time.time()
        for j in range(n_train_batches):
            batch_x = train_x[j*batch_size:(j+1)*batch_size]
            batch_y = train_y[j*batch_size:(j+1)*batch_size]
            _, batch_cent, batch_acc = sess.run([train_op, cent, acc], 
                    {x:batch_x, y:batch_y, is_training:True})
            train_cent += batch_cent
            train_acc += batch_acc
        train_cent /= n_train_batches
        train_acc /= n_train_batches

        valid_cent = 0.
        valid_acc = 0.
        for j in range(n_valid_batches):
            batch_x = valid_x[j*batch_size:(j+1)*batch_size]
            batch_y = valid_y[j*batch_size:(j+1)*batch_size]
            batch_cent, batch_acc = sess.run([cent, acc], 
                    {x:batch_x, y:batch_y, is_training:False})
            valid_cent += batch_cent
            valid_acc += batch_acc
        valid_cent /= n_valid_batches
        valid_acc /= n_valid_batches

        line = 'epoch %d (%f), train cent %f, train acc %f, valid cent %f, valid acc %f' \
                % (i, time.time()-start, train_cent, train_acc, valid_cent, valid_acc)
        print line
        logfile.write(line + '\n')

    logfile.close()
    saver.save(sess, FLAGS.save_dir+'/model.ckpt')

from utils.image import batchmat_to_tileimg
def test():
    saver.restore(sess, FLAGS.save_dir+'/model.ckpt')
    batch_x = valid_x[0:10]
    batch_code = sess.run(code, {x:batch_x, is_training:False})
    print batch_code
    plt.figure()
    plt.imshow(batchmat_to_tileimg(batch_x, (56, 56), (1, 10)))
    plt.gray()
    plt.axis('off')
    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
