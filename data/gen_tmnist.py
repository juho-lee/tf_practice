import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cPickle as pkl
import gzip
import os

mnist = input_data.read_data_sets('mnist')
width = 50
lim = width - 28 + 1

def translate(x):
    y = np.zeros((width, width))
    h = np.random.randint(lim)
    w = np.random.randint(lim)
    y[h:h+28, w:w+28] = x.reshape(28, 28)
    return y.reshape(width*width)

n_train = mnist.train.num_examples
print 'number of train samples: %d' % n_train
train_x = mnist.train.images
train_y = mnist.train.labels
train_tx = np.zeros((n_train, width*width))
for i in range(n_train):
    train_tx[i,:] = translate(train_x[i,:])  
del train_x

n_valid = mnist.validation.num_examples
print 'number of validation samples: %d' % n_valid
valid_x = mnist.validation.images
valid_y = mnist.validation.labels
valid_tx = np.zeros((n_valid, width*width))
for i in range(n_valid):
    valid_tx[i,:] = translate(valid_x[i,:])
del valid_x

n_test = mnist.test.num_examples
print 'number of test samples: %d' % n_test
test_x = mnist.test.images
test_y = mnist.test.labels
test_tx = np.zeros((n_test, width*width))
for i in range(n_test):
    test_tx[i,:] = translate(test_x[i,:])
del test_x

output_dir = 'tmnist'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

f = gzip.open(output_dir + '/tmnist.pkl.gz', 'wb')
pkl.dump([(train_tx, train_y), (valid_tx, valid_y), 
    (test_tx, test_y)], f, protocol=2)
f.close()



