import tensorflow as tf
fc = tf.contrib.layers.fully_connected
from prob import *
from tensorflow.examples.tutorials.mnist import input_data
import time
from utils.image import batchmat_to_tileimg, gen_grid
import matplotlib.pyplot as plt
import seaborn as sns

n_hid = 500
n_lat = 20
n_fac = 10

x = tf.placeholder(tf.float32, shape=[None, 784])
h_enc = fc(x, n_hid)
z_mean = fc(h_enc, n_lat, activation_fn=None)
z_log_var = fc(h_enc, n_lat, activation_fn=None)
z = gaussian_sample(z_mean, z_log_var)
w_mean = fc(h_enc, n_fac, activation_fn=None)
w_log_var = fc(h_enc, n_fac, activation_fn=None)
w = rect_gaussian_sample(w_mean, w_log_var)

h_dec = fc(z, n_hid)
P = fc(h_dec, n_fac*784, activation_fn=None)
p = tf.slice(w, [0,0], [-1,1]) * tf.slice(P, [0,0], [-1,784])
for i in range(1, n_fac):
    p = p + tf.slice(w, [0,i], [-1,1]) * tf.slice(P, [0,784*i], [-1,784])
p = tf.nn.sigmoid(p)

neg_ll = bernoulli_neg_ll(x, p)
kld_w = rect_gaussian_kld(w_mean, w_log_var, mean0=-1.)
kld_z = gaussian_kld(z_mean, z_log_var)
loss = neg_ll + kld_w + kld_z
train_step = tf.train.AdamOptimizer().minimize(loss)

mnist = input_data.read_data_sets("data/mnist")
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_test_batches = mnist.test.num_examples / batch_size

n_epochs = 20
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(n_epochs):
        start = time.time()
        train_neg_ll = 0.
        train_kld_w = 0.
        train_kld_z = 0.
        for j in range(n_train_batches):
            batch_x, _ = mnist.train.next_batch(batch_size)
            _, batch_neg_ll, batch_kld_w, batch_kld_z = \
                    sess.run([train_step, neg_ll, kld_w, kld_z],
                            feed_dict={x:batch_x})
            train_neg_ll += batch_neg_ll
            train_kld_w += batch_kld_w
            train_kld_z += batch_kld_z
        train_neg_ll /= n_train_batches
        train_kld_w /= n_train_batches
        train_kld_z /= n_train_batches

        test_neg_ll = 0.
        test_kld_w = 0.
        test_kld_z = 0.
        for j in range(n_test_batches):
            batch_x, _ = mnist.test.next_batch(batch_size)
            batch_neg_ll, batch_kld_w, batch_kld_z = \
                    sess.run([neg_ll, kld_w, kld_z], feed_dict={x:batch_x})
            test_neg_ll += batch_neg_ll
            test_kld_w += batch_kld_w
            test_kld_z += batch_kld_z
        test_neg_ll /= n_test_batches
        test_kld_w /= n_test_batches
        test_kld_z /= n_test_batches

        print "Epoch %d (%f sec), train loss %f = %f + %f + %f, test loss %f = %f + %f + %f" \
                % (i+1, time.time()-start,
                    train_neg_ll+train_kld_w+train_kld_z,
                    train_neg_ll, train_kld_w, train_kld_z,
                    test_neg_ll+test_kld_w+train_kld_z,
                    test_neg_ll, test_kld_w, test_kld_z)

    test_x, test_y = mnist.test.next_batch(100)
    plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(test_x, (28, 28), (10, 10)))

    plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    p_recon = sess.run(p, {x:test_x})
    plt.imshow(batchmat_to_tileimg(p_recon, (28, 28), (10, 10)))

    test_w = np.zeros((n_fac*n_fac, n_fac))
    for i in range(n_fac):
        test_w[i*n_fac:(i+1)*n_fac, i] = 1.0
    test_z = np.random.normal(size=(n_fac*n_fac, n_lat))
    p_gen = sess.run(p, {w:test_w, z:test_z})
    I_gen = batchmat_to_tileimg(p_gen, (28, 28), (n_fac, n_fac))
    plt.figure('generated')
    plt.gray()
    plt.axis('off')
    plt.imshow(I_gen)

    plt.figure('factor activation heatmap')
    hist = np.zeros((10, n_fac))
    for i in range(n_test_batches):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        batch_w = sess.run(w, {x:batch_x})
        for i in range(batch_size):
            hist[batch_y[i], batch_w[i] > 0] += 1
    sns.heatmap(hist)

    plt.show()
