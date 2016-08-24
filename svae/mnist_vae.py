import tensorflow as tf
fc = tf.contrib.layers.fully_connected
from prob import *
from tensorflow.examples.tutorials.mnist import input_data
import time
from utils.image import batchmat_to_tileimg, gen_grid
import matplotlib.pyplot as plt

n_hid = 200
n_lat = 20
x = tf.placeholder(tf.float32, shape=[None, 784])
h_enc = fc(x, n_hid)
z_mean = fc(h_enc, n_lat, activation_fn=None)
z_log_var = fc(h_enc, n_lat, activation_fn=None)
z = gaussian_sample(z_mean, z_log_var)
h_dec = fc(z, n_hid)
p = fc(h_dec, 784, activation_fn=tf.nn.sigmoid)

neg_ll = bernoulli_neg_ll(x, p)
kld = gaussian_kld(z_mean, z_log_var)
loss = neg_ll + kld
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
        train_kld = 0.
        for j in range(n_train_batches):
            batch_x, _ = mnist.train.next_batch(batch_size)
            _, batch_neg_ll, batch_kld = \
                    sess.run([train_step, neg_ll, kld], feed_dict={x:batch_x})
            train_neg_ll += batch_neg_ll
            train_kld += batch_kld
        train_neg_ll /= n_train_batches
        train_kld /= n_train_batches

        test_neg_ll = 0.
        test_kld = 0.
        for j in range(n_test_batches):
            batch_x, _ = mnist.test.next_batch(batch_size)
            batch_neg_ll, batch_kld = \
                    sess.run([neg_ll, kld], feed_dict={x:batch_x})
            test_neg_ll += batch_neg_ll
            test_kld += batch_kld
        test_neg_ll /= n_test_batches
        test_kld /= n_test_batches

        print "Epoch %d (%f sec), train loss %f = %f + %f, test loss %f = %f + %f" \
                % (i+1, time.time()-start,
                    train_neg_ll+train_kld, train_neg_ll, train_kld,
                    test_neg_ll+test_kld, test_neg_ll, test_kld)

    test_x, _ = mnist.test.next_batch(100)
    plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(test_x, (28, 28), (10, 10)))

    plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    p_recon = sess.run(p, {x:test_x})
    plt.imshow(batchmat_to_tileimg(p_recon, (28, 28), (10, 10)))

    plt.figure('generated')
    plt.gray()
    plt.axis('off')
    eps = np.zeros((10*10, n_lat))
    for i in range(10):
        eps[i*10:(i+1)*10, i*2:(i+1)*2] = 5*np.random.normal(size=(10, 2))
    p_gen = sess.run(p, {z:eps})
    plt.imshow(batchmat_to_tileimg(p_gen, (28, 28), (10, 10)))

    plt.show()
