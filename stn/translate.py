import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

def repeat(x, n):
    return tf.reshape(tf.tile(tf.expand_dims(x, 1), tf.pack([1,n])), [-1])

def translate(U, theta, out_height, out_width):
    num_batch = tf.shape(U)[0]
    height, width, num_ch = U.get_shape()[1:]
    height = height.value
    width = width.value
    num_ch = num_ch.value
    hwc = height*width*num_ch

    nind = tf.range(num_batch)
    x = repeat(tf.range(height), width)
    y = tf.tile(tf.range(width), tf.pack([height]))
    cind = tf.range(num_ch)

    nind = tf.expand_dims(repeat(nind, hwc), 1)
    x = tf.tile(tf.expand_dims(repeat(x, num_ch), 1), tf.pack([num_batch,1]))
    y = tf.tile(tf.expand_dims(repeat(y, num_ch), 1), tf.pack([num_batch,1]))
    cind = tf.tile(tf.expand_dims(cind, 1), tf.pack([num_batch*height*width,1]))

    dx, dy = tf.split(1, 2, theta)
    dx = tf.cast(tf.clip_by_value(dx, 0, out_height-height), 'int32')
    dx = tf.reshape(tf.tile(dx, tf.pack([1,hwc])), [-1,1])
    dy = tf.cast(tf.clip_by_value(dy, 0, out_width-width), 'int32')
    dy = tf.reshape(tf.tile(dy, tf.pack([1,hwc])), [-1,1])
    x = x + dx
    y = y + dy

    tind = tf.concat(1, [nind, x, y, cind])
    val = tf.reshape(U, [-1])
    T = tf.sparse_to_dense(tind,
            tf.pack([num_batch, out_height, out_width, num_ch]),
            val)
    T.set_shape([None, out_height, out_width, num_ch])
    return T

def to_trans(input):
    if len(input.get_shape()) == 4:
        input = layers.flatten(input)
    num_inputs = input.get_shape()[1]
    W_init = tf.constant_initializer(np.zeros((num_inputs, 2)))
    b_init = tf.constant_initializer(np.array([0.,0.]))
    return layers.fully_connected(input, 2,
            weights_initializer=W_init,
            biases_initializer=b_init)

if __name__ == '__main__':
    from scipy import ndimage
    import matplotlib.pyplot as plt
    U = ndimage.imread('gong13.jpg')
    height, width, num_ch = U.shape
    U = U / 255.
    U = U.reshape(1, height, width, num_ch).astype('float32')

    T_true = ndimage.imread('gong_trans.jpg')
    out_height, out_width, num_ch = T_true.shape
    T_true = T_true / 255.
    T_true = T_true.reshape(1, out_height, out_width, num_ch).astype('float32')

    U_ = tf.placeholder(tf.float32, [1, height, width, num_ch])
    T_true_ = tf.placeholder(tf.float32, [1, out_height, out_width, num_ch])
    theta = tf.Variable(tf.zeros([1, 2]))
    T_ = translate(U_, theta, out_height, out_width)
    loss = tf.reduce_sum(tf.square(T_true_ - T_))

    train_op = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for i in range(10):
        _, l = sess.run([train_op, loss], {U_:U, T_true_:T_true})
        print l


    """
    U_ = tf.placeholder(tf.float32, [None, height, width, num_ch])
    theta_ = tf.placeholder(tf.float32, [None, 2])
    T_ = translate(U_, theta_, height*2, width*2)

    theta = np.array([[200, 300], [4000, 1000]])
    sess = tf.Session()
    T = sess.run(T_, {U_:np.tile(U, [2, 1, 1, 1]), theta_:theta})

    plt.figure()
    plt.imshow(T[0])
    plt.show()

    from PIL import Image
    I = Image.fromarray((T[0]*255).astype(np.uint8))
    I.save('gong_trans.jpg')
    """

