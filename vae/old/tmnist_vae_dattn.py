import tensorflow as tf
fc = tf.contrib.layers.fully_connected
flat = tf.contrib.layers.flatten
from prob import *
import time
from utils.data import load_pkl
from utils.image import batchmat_to_tileimg
import matplotlib.pyplot as plt
from stn.draw_attn import *

height = 50
width = 50
N = 30
n_hid = 800
n_lat = 30

attunit = AttentionUnit(height, width, 1, N)
x = tf.placeholder(tf.float32, [None, height*width])
att_enc = fc(fc(fc(x, n_hid/10), n_hid), 5, activation_fn=None)
x_att = attunit.read(x, att_enc)
hid_enc = fc(tf.concat(1, [att_enc, x_att]), n_hid)
z_mean = fc(hid_enc, n_lat, activation_fn=None)
z_log_var = fc(hid_enc, n_lat, activation_fn=None)
z = gaussian_sample(z_mean, z_log_var)

hid_dec = fc(z, n_hid)
att_dec = fc(fc(hid_dec, n_hid/10), 5, activation_fn=None)
p_att = fc(hid_dec, N*N, activation_fn=None)
p = tf.nn.sigmoid(attunit.write(p_att, att_dec))

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

n_epochs = 50
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
plt.imshow(batchmat_to_tileimg(test_x[0:100], (height, width), (10, 10)))

plt.figure('reconstructed')
plt.gray()
plt.axis('off')
recon_att, recon = sess.run([tf.nn.sigmoid(p_att), p], {x:test_x[0:100]})
plt.imshow(batchmat_to_tileimg(recon, (height, width), (10, 10)))

plt.figure('reconstructed_attended')
plt.gray()
plt.axis('off')
plt.imshow(batchmat_to_tileimg(recon_att, (N, N), (10, 10)))

plt.figure('attended')
plt.gray()
plt.axis('off')
att = sess.run(x_att, {x:test_x[0:100]})
plt.imshow(batchmat_to_tileimg(att, (N, N), (10, 10)))

plt.figure('generated')
plt.gray()
plt.axis('off')
gen_att, gen = sess.run([tf.nn.sigmoid(p_att), p], {z:np.random.normal(size=(100, n_lat))})
plt.imshow(batchmat_to_tileimg(gen, (height, width), (10, 10)))

plt.figure('generated_attended')
plt.gray()
plt.axis('off')
plt.imshow(batchmat_to_tileimg(gen_att, (N, N), (10, 10)))

plt.show()

