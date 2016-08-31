import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cPickle as pkl
import gzip
import os

mnist = input_data.read_data_sets('mnist')
width = 60
lim = width - 28 + 1
nf_min = 2
nf_max = 4

def make_multi_mnist(x, n):
    mx = np.zeros((n, width*width))
    c = np.zeros(n)
    offset = 0
    for i in range(n):
        n_fig = np.random.randint(nf_min, nf_max)
        y = np.zeros((width, width))
        for j in range(n_fig):
            if offset + j >= len(x):
                offset = 0
                np.random.shuffle(x)
            h = int((lim-1)/(1 + np.exp(np.random.normal(0,100))))
            w = int((lim-1)/(1 + np.exp(np.random.normal(0,100))))
            y[h:h+28,w:w+28] = x[offset+j].reshape(28, 28)
        mx[i,:] = y.reshape(width*width)
        c[i] = n_fig
        offset += n_fig
    return mx, c

n_train = 30000
print 'number of train samples: %d' % n_train
train_mx, train_c = make_multi_mnist(mnist.train.images, n_train)

n_valid = 2000
print 'number of validation samples: %d' % n_valid
valid_mx, valid_c = make_multi_mnist(mnist.validation.images, n_valid)

n_test = 3000
print 'number of test samples: %d' % n_test
test_mx, test_c = make_multi_mnist(mnist.test.images, n_test)

output_dir = 'mmnist'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

f = gzip.open(output_dir + '/mmnist.pkl.gz', 'wb')
pkl.dump([(train_mx, train_c), (valid_mx, valid_c), (test_mx, test_c)], f, protocol=2)
f.close()

