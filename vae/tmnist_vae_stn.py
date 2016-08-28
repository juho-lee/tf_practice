import tensorflow as tf
fc = tf.contrib.layers.fully_connected
flat = tf.contrib.layers.flatten
from prob import *
import time
from utils.data import load_pkl
from utils.image import batchmat_to_tileimg, batchimg_to_tileimg
import matplotlib.pyplot as plt
from stn.spatial_transformer import spatial_transformer, loc_last

h = 50
w = 50
n_hid = 800
n_lat = 20
h_att = 20
w_att = 20

x = tf.placeholder(tf.float32, [None, h*w])
x_img = tf.reshape(x, [-1, h, w, 1])
loc_enc = loc_last(fc(fc(x, n_hid), n_hid/10))
x_att = spatial_transformer(x_img, loc_enc, [h_att, w_att])
h_enc = fc(tf.concat(1, [loc_enc, flat(x_att)]), n_hid)
z_mean = fc(h_enc, n_lat, activation_fn=None)
z_log_var = fc(h_enc, n_lat, activation_fn=None)
z = gaussian_sample(z_mean, z_log_var)

h_dec = fc(z, n_hid)
loc_dec = loc_last(fc(h_dec, n_hid/10))

"""
p_att = tf.reshape(fc(h_dec, h_att*w_att, activation_fn=tf.nn.tanh), [-1, h_att, w_att, 1])
p = flat(tf.nn.sigmoid(spatial_transformer(p_att, loc_dec, [h, w])))
"""
p_att = tf.reshape(fc(h_dec, h_att*w_att, activation_fn=tf.nn.sigmoid), [-1, h_att, w_att, 1])
p = tf.clip_by_value(spatial_transformer(p_att, loc_dec, [h, w]), 0, 1)
p = flat(p)

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

n_epochs = 1
sess = tf.Session()
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
recon_att, recon = sess.run([p_att, p], {x:test_x[0:100]})
plt.imshow(batchmat_to_tileimg(recon, (h, w), (10, 10)))

plt.figure('reconstructed_attended')
plt.gray()
plt.axis('off')
plt.imshow(batchimg_to_tileimg(recon_att, (10, 10)))

plt.figure('attended')
plt.gray()
plt.axis('off')
att = sess.run(x_att, {x:test_x[0:100]})
plt.imshow(batchimg_to_tileimg(att, (10, 10)))

plt.figure('generated')
plt.gray()
plt.axis('off')
gen_att, gen = sess.run([p_att, p], {z:np.random.normal(size=(100, n_lat))})
plt.imshow(batchmat_to_tileimg(gen, (h, w), (10, 10)))

plt.figure('generated_attended')
plt.gray()
plt.axis('off')
plt.imshow(batchimg_to_tileimg(gen_att, (10, 10)))

plt.show()

