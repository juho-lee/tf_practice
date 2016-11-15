import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cPickle as pkl
import gzip
import os
from scipy.misc import imresize

mnist = input_data.read_data_sets('mnist')

#swidth = 14
#width = 32
swidth = 28
width = 56

lim = width - swidth + 1
min_intval = 20

def sample_offset():
    h = np.random.randint(0, lim)
    w = np.random.randint(0, lim)
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
        while (((h0-h1) < min_intval) & ((w0-w1) < min_intval)):
            h0, w0 = sample_offset()
            h1, w1 = sample_offset()

        src = x[offset].reshape(swidth, swidth)
        t[h0:h0+swidth, w0:w0+swidth] += src

        src = x[offset+1].reshape(swidth, swidth)
        t[h1:h1+swidth, w1:w1+swidth] += src

        t = np.clip(t, 0., 1.)            

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

train_x = mnist.train.images
train_y = mnist.train.labels

dx, dy = make_double_mnist(train_x, train_y, 60000)
train_ind = (dy <= 13)

train_dx = dx[train_ind]
train_dy = dy[train_ind]
n_train = len(train_dx)
n_test = int(0.1*n_train)
n_train -= n_test
test_dx = train_dx[n_train:]
test_dy = train_dy[n_train:]
train_dx = train_dx[0:n_train]
train_dy = train_dy[0:n_train]

# make one shot data: with 7, 8, 9
# class0: 7 + 7 = 14
# class1: 7 + 8 = 15
# class2: 7 + 9 = 16
# class3: 8 + 9 = 17
# class4: 9 + 9 = 18
ind7 = train_y == 7
ind8 = train_y == 8
ind9 = train_y == 9

ind = ind7
one_shot_dx, one_shot_dy = make_double_mnist(train_x[ind], train_y[ind], 1000)

ind = ind7 | ind8
tmp_x, tmp_y = make_double_mnist(train_x[ind], train_y[ind], 1000)
one_shot_dx = np.concatenate([one_shot_dx, tmp_x], 0)
one_shot_dy = np.concatenate([one_shot_dy, tmp_y], 0)

ind = ind7 | ind9
tmp_x, tmp_y = make_double_mnist(train_x[ind], train_y[ind], 1000)
one_shot_dx = np.concatenate([one_shot_dx, tmp_x], 0)
one_shot_dy = np.concatenate([one_shot_dy, tmp_y], 0)

ind = ind8 | ind9
tmp_x, tmp_y = make_double_mnist(train_x[ind], train_y[ind], 1000)
one_shot_dx = np.concatenate([one_shot_dx, tmp_x], 0)
one_shot_dy = np.concatenate([one_shot_dy, tmp_y], 0)

ind = ind9
tmp_x, tmp_y = make_double_mnist(train_x[ind], train_y[ind], 1000)
one_shot_dx = np.concatenate([one_shot_dx, tmp_x], 0)
one_shot_dy = np.concatenate([one_shot_dy, tmp_y], 0)


print 'number of train samples: %d' % len(train_dx)
print 'number of test samples: %d' % len(test_dx)
print 'number of one-shot samples: %d' % len(one_shot_dx)

from utils.image import batchmat_to_tileimg
import matplotlib.pyplot as plt
print train_dy[0:20]
plt.figure()
plt.imshow(batchmat_to_tileimg(train_dx[0:20], (width, width), (4, 5)))
plt.gray()
plt.axis('off')
plt.show()

print
print one_shot_dy[0:20]
plt.figure()
plt.imshow(batchmat_to_tileimg(one_shot_dx[0:20], (width, width), (4, 5)))
plt.gray()
plt.axis('off')
plt.show()


output_dir = 'dmnist'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

f = gzip.open(output_dir + '/dmnist.pkl.gz', 'wb')
pkl.dump([(train_dx, train_dy), (test_dx, test_dy), (one_shot_dx, one_shot_dy)], f, protocol=2)
f.close()


