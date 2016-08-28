import tensorflow as tf
fc = tf.contrib.layers.fully_connected
flat = tf.contrib.layers.flatten
from prob import *
from tensorflow.examples.tutorials.mnist import input_data
import time
from stn.spatial_transformer import spatial_transformer, loc_last
from utils.image import batchmat_to_tileimg, gen_grid
import matplotlib.pyplot as plt
import seaborn as sns

n_hid = 400
n_lat = 20
n_fac = 5

x = tf.placeholder(tf.float32, shape=[None, 28*28])
x_img = tf.reshape(x, [-1, 28, 28, 1])
h_enc_att = fc(fc(x, n_hid), n_hid/10)
loc_enc = loc_last(h_enc_att)
x_att = flat(spatial_transformer(x_img, loc_enc, [5, 5]))
for i in range(1, n_fac):
    loc_enc = loc_last(h_enc_att)
    x_att = tf.concat(1, 
        [x_att, flat(spatial_transformer(x_img, loc_enc, [5, 5]))])
h_enc = fc(tf.concat(1, [loc_enc, x_att]), n_hid)
z_mean = fc(h_enc, n_lat, activation_fn=None)
z_log_var = fc(h_enc, n_lat, activation_fn=None)
z = gaussian_sample(z_mean, z_log_var)
w_mean = fc(h_enc, n_fac, activation_fn=None)
w_log_var = fc(h_enc, n_fac, activation_fn=None)
w = rect_gaussian_sample(w_mean, w_log_var)

h_dec = fc(z, n_hid)
h_dec_att = fc(h_dec, n_hid/10)
loc_dec = loc_last(h_dec_att)
p_att = tf.slice(w, [0,0], [-1,1]) * fc(h_dec, 5*5, activation_fn=None)
p_att = tf.reshape(tf.nn.sigmoid(p_att), [-1, 5, 5, 1])
p = spatial_transformer(p_att, loc_dec, [28, 28])
for i in range(1, n_fac):
    loc_dec = loc_last(h_dec_att)
    p_att = tf.slice(w, [0,i], [-1,1]) * fc(h_dec, 5*5, activation_fn=None)
    p_att = tf.reshape(tf.nn.sigmoid(p_att), [-1, 5, 5, 1])
    p = p + spatial_transformer(p_att, loc_dec, [28, 28])
p = flat(tf.clip_by_value(p, 0, 1))

neg_ll = bernoulli_neg_ll(x, p)
kld_w = rect_gaussian_kld(w_mean, w_log_var, mean0=-1.)
kld_z = gaussian_kld(z_mean, z_log_var)
loss = neg_ll + kld_w + kld_z
train_step = tf.train.AdamOptimizer().minimize(loss)

mnist = input_data.read_data_sets("data/mnist")
batch_size = 100
n_train_batches = mnist.train.num_examples / batch_size
n_test_batches = mnist.test.num_examples / batch_size

n_epochs = 30
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
