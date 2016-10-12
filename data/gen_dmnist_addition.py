import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cPickle as pkl
import gzip
import os

mnist = input_data.read_data_sets('mnist')
width = 56
lim = width - 28 + 1

def sample_offset():
    h = int((lim-1)/(1+np.exp(np.random.normal(0,1))))
    w = int((lim-1)/(1+np.exp(np.random.normal(0,1))))
    return h, w

def make_double_mnist(x, y, n):
    dx = np.zeros((n, width*width))
    dy = np.zeros((n,))
    offset = 0
    for i in range(n):
        if (i%100) == 0:
            print i
        t = np.zeros((width, width))
        h0, w0 = sample_offset()
        h1, w1 = sample_offset()
        while (((h0-h1) < 10) & ((w0-w1) < 10)):
            h0, w0 = sample_offset()
            h1, w1 = sample_offset()

        src = x[offset].reshape(28, 28)
        ind = np.where(src>0)
        t[h0+ind[0],w0+ind[1]] = src[ind[0],ind[1]]

        src = x[offset+1].reshape(28, 28)
        ind = np.where(src>0)
        t[h1+ind[0],w1+ind[1]] = src[ind[0],ind[1]]

        dx[i,:] = t.reshape(width*width)
        dy[i] = y[offset] + y[offset+1]
        offset += 2
        if offset >= len(x):
            offset = 0
            ind = range(len(x))
            np.random.shuffle(ind)
            x = x[ind]
            y = y[ind]
    return dx, dy

dx, dy = make_double_mnist(mnist.train.images,
        mnist.train.labels, 60000)
train_ind = (dy <= 14)

train_dx = dx[train_ind]
train_dy = dy[train_ind]
n_train = len(train_dx)
n_valid = int(0.1*n_train)
n_train -= n_valid
valid_dx = train_dx[n_train:]
valid_dy = train_dy[n_train:]
train_dx = train_dx[0:n_train]
train_dy = train_dy[0:n_train]
test_ind = (dy > 14)
test_dx = dx[test_ind]
test_dy = dy[test_ind]

print 'number of train samples: %d' % len(train_dx)
print 'number of validation samples: %d' % len(valid_dx)
print 'number of test samples: %d' % len(test_dx)

output_dir = 'dmnist'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

f = gzip.open(output_dir + '/dmnist_addition.pkl.gz', 'wb')
pkl.dump([(train_dx, train_dy), (valid_dx, valid_dy), (test_dx, test_dy)], f, protocol=2)
f.close()

from utils.image import batchmat_to_tileimg
import matplotlib.pyplot as plt
print train_dy[0:20]
plt.figure()
plt.imshow(batchmat_to_tileimg(train_dx[0:20], (width, width), (4, 5)))
plt.gray()
plt.axis('off')
plt.show()

