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
N = 30
n_hid = 800
n_lat = 30
n_fac = 10

attunit = AttentionUnit(height, width, 1, N)
x = tf.placeholder(tf.float32, [None, height*width])
att_enc = fc(fc(fc(x, n_hid/10), n_hid), 5*n_fac, activation_fn=None)
x_att = attunit.read_multiple(x, att_enc, n_fac)
hid_enc = fc(tf.concat(1, [att_enc, x_att]), n_hid)
z_mean = fc(hid_enc, n_lat, activation_fn=None)
z_log_var = fc(hid_enc, n_lat, activation_fn=None)
z = gaussian_sample(z_mean, z_log_var)
w_mean = fc(hid_enc, n_fac, activation_fn=None)
w_log_var = fc(hid_enc, n_fac, activation_fn=None)
w = rect_gaussian_sample(w_mean, w_log_var)

hid_dec = fc(z, n_hid)
att_dec = fc(fc(hid_dec, n_hid/10), 5*n_fac, activation_fn=None)
p_att = fc(hid_dec, N*N*n_fac, activation_fn=None)
P = attunit.write_multiple(p_att, att_dec, n_fac)
p = tf.slice(w, [0,0], [-1,1])*tf.slice(P, [0,0], [-1,height*width])
for i in range(1, n_fac):
    p = p + tf.slice(w, [0,i], [-1,1])*\
            tf.slice(P, [0,height*width*i], [-1,height*width])
p = tf.nn.sigmoid(p)

neg_ll = bernoulli_neg_ll(x, p)
kld_w = rect_gaussian_kld(w_mean, w_log_var)
kld_z = gaussian_kld(z_mean, z_log_var)
loss = neg_ll + kld_w + kld_z
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
        train_kld_w = 0.
        train_kld_z = 0.
        for j in range(n_train_batches):
            batch_x = train_x[j*batch_size:(j+1)*batch_size]
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
            batch_x = test_x[j*batch_size:(j+1)*batch_size]
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

    test_w = np.zeros((n_fac*n_fac, n_fac))
    for i in range(n_fac):
        test_w[i*n_fac:(i+1)*n_fac, i] = 1.0
    test_z = np.random.normal(size=(n_fac*n_fac, n_lat))
    p_gen = sess.run(p, {w:test_w, z:test_z})
    I_gen = batchmat_to_tileimg(p_gen, (height, width), (n_fac, n_fac))
    plt.figure('generated')
    plt.gray()
    plt.axis('off')
    plt.imshow(I_gen)

    plt.show()
