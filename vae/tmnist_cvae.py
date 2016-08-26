import tensorflow as tf
fc = tf.contrib.layers.fully_connected
conv = tf.contrib.layers.convolution2d
pool = tf.contrib.layers.max_pool2d
deconv = tf.contrib.layers.convolution2d_transpose
flat = tf.contrib.layers.flatten

from prob import *
import time
from utils.data import load_pkl
from utils.image import batchmat_to_tileimg
import matplotlib.pyplot as plt

h = 50
w = 50
n_lat = 20

x = tf.placeholder(tf.float32, [None, h*w])
x_img = tf.reshape(x, [-1, h, w, 1])
h_enc = pool(conv(x_img, 4, [3, 3]), [2, 2])
h_enc = pool(conv(h_enc, 8, [3, 3]), [2, 2])
h_enc = pool(conv(h_enc, 16, [3, 3]), [2, 2])
h_enc = pool(conv(h_enc, 32, [3, 3]), [2, 2])
h_enc = flat(h_enc)
z_mean = fc(h_enc, n_lat, activation_fn=None)
z_log_var = fc(h_enc, n_lat, activation_fn=None)
z = gaussian_sample(z_mean, z_log_var)
h_dec = tf.reshape(fc(z, 32*3*3), [-1, 3, 3, 32])
h_dec = deconv(h_dec, 16, [2, 2], [2, 2])
h_dec = deconv(h_dec, 8, [2, 2], [2, 2])
h_dec = deconv(h_dec, 4, [3, 3], [2, 2], padding='VALID')
p = flat(deconv(h_dec, 1, [2, 2], [2, 2], activation_fn=tf.nn.sigmoid))

neg_ll = bernoulli_neg_ll(x, p)
kld = gaussian_kld(z_mean, z_log_var)
loss = neg_ll + kld
train_step = tf.train.AdamOptimizer().minimize(loss)

train_xy, _, test_xy = load_pkl('data/tmnist/tmnist.pkl.gz')
train_x, _ = train_xy
test_x, _ = test_xy
batch_size = 100
n_train_batches = len(train_x) / batch_size
n_test_batches = len(test_x) / batch_size

n_epochs = 30
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(n_epochs):
        start = time.time()
        train_neg_ll = 0.
        train_kld = 0.
        for j in range(n_train_batches):
            batch_x = train_x[j*batch_size:(j+1)*batch_size]
            _, batch_neg_ll, batch_kld = \
                    sess.run([train_step, neg_ll, kld], {x:batch_x})
            train_neg_ll += batch_neg_ll
            train_kld += batch_kld
        train_neg_ll /= n_train_batches
        train_kld /= n_train_batches

        test_neg_ll = 0.
        test_kld = 0.
        for j in range(n_test_batches):
            batch_x = test_x[j*batch_size:(j+1)*batch_size]
            batch_neg_ll, batch_kld = sess.run([neg_ll, kld], {x:batch_x})
            test_neg_ll += batch_neg_ll
            test_kld += batch_kld
        test_neg_ll /= n_test_batches
        test_kld /= n_test_batches

        print "Epoch %d (%f sec), train loss %f = %f + %f, test loss %f = %f + %f" \
                % (i+1, time.time()-start,
                    train_neg_ll+train_kld, train_neg_ll, train_kld,
                    test_neg_ll+test_kld, test_neg_ll, test_kld)

    plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(test_x[0:100], (h, w), (10, 10)))

    plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    p_recon = sess.run(p, {x:test_x[0:100]})
    plt.imshow(batchmat_to_tileimg(p_recon, (h, w), (10, 10)))

    plt.figure('generated')
    plt.gray()
    plt.axis('off')
    p_gen = sess.run(p, {z:np.random.normal(size=(100, n_lat))})
    plt.imshow(batchmat_to_tileimg(p_gen, (h, w), (10, 10)))

    plt.show()
