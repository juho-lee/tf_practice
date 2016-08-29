import tensorflow as tf
fc = tf.contrib.layers.fully_connected
flat = tf.contrib.layers.flatten
from prob import *
import time
from utils.data import load_pkl
from utils.image import batchmat_to_tileimg
import matplotlib.pyplot as plt
from stn.draw_attn import *

height = 60
width = 60
n_hid = 800
n_lat = 30

x = tf.placeholder(tf.float32, [None, height*width])
hid_enc = fc(x, n_hid)
z_mean = fc(hid_enc, n_lat, activation_fn=None)
z_log_var = fc(hid_enc, n_lat, activation_fn=None)
z = gaussian_sample(z_mean, z_log_var)
hid_dec = fc(z, n_hid)
p = fc(hid_dec, height*width, activation_fn=tf.nn.sigmoid)

neg_ll = bernoulli_neg_ll(x, p)
kld = gaussian_kld(z_mean, z_log_var)
loss = neg_ll + kld
train_step = tf.train.AdamOptimizer().minimize(loss)

train_x, _, test_x = load_pkl('data/mmnist/mmnist.pkl.gz')
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
                    sess.run([train_step, neg_ll, kld], feed_dict={x:batch_x})
            train_neg_ll += batch_neg_ll
            train_kld += batch_kld
        train_neg_ll /= n_train_batches
        train_kld /= n_train_batches

        test_neg_ll = 0.
        test_kld = 0.
        for j in range(n_test_batches):
            batch_x = test_x[j*batch_size:(j+1)*batch_size]
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

    batch_x = test_x[0:100]
    plt.figure('original')
    plt.gray()
    plt.axis('off')
    plt.imshow(batchmat_to_tileimg(batch_x, (height, width), (10, 10)))

    plt.figure('reconstructed')
    plt.gray()
    plt.axis('off')
    p_recon = sess.run(p, {x:batch_x})
    plt.imshow(batchmat_to_tileimg(p_recon, (height, width), (10, 10)))

    p_gen = sess.run(p, {z:np.random.normal(size=(100, n_lat))})
    I_gen = batchmat_to_tileimg(p_gen, (height, width), (10, 10))
    plt.figure('generated')
    plt.gray()
    plt.axis('off')
    plt.imshow(I_gen)

    plt.show()
