import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cPickle as pkl
import gzip
import os

mnist = input_data.read_data_sets('mnist')
width = 56
lim = width - 28 + 1

def make_double_mnist(x, y, n):
    dx = np.zeros((n, width*width))
    dy = np.zeros((n,))
    offset = 0
    for i in range(n):
        t = np.zeros((width, width))

        """
        for j in range(2):
            h = int((lim-1)/(1 + np.exp(np.random.normal(0,50))))
            w = int((lim-1)/(1 + np.exp(np.random.normal(0,50))))
            y[h:h+28,w:w+28] = x[offset+j].reshape(28, 28)
        """

        t[0:28,0:28] = x[offset].reshape(28, 28)
        t[28:,28:] = x[offset+1].reshape(28, 28)

        dx[i,:] = t.reshape(width*width)
        dy[i] = y[offset]
        offset += 2
        if offset >= len(x):
            offset = 0
            ind = range(len(x))
            np.random.shuffle(ind)
            x = x[ind]
            y = y[ind]

    return dx, dy

n_train = 30000
print 'number of train samples: %d' % n_train
train_dx, train_dy = make_double_mnist(mnist.train.images, 
        mnist.train.labels, n_train)

n_valid = 2000
print 'number of validation samples: %d' % n_valid
valid_dx, valid_dy = make_double_mnist(mnist.validation.images, 
        mnist.validation.labels, n_valid)

n_test = 3000
print 'number of test samples: %d' % n_test
test_dx, test_dy = make_double_mnist(mnist.test.images, 
        mnist.test.labels, n_test)

output_dir = 'dmnist'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

f = gzip.open(output_dir + '/dmnist.pkl.gz', 'wb')
pkl.dump([(train_dx, train_dy), (valid_dx, valid_dy), (test_dx, test_dy)], f, protocol=2)
f.close()



