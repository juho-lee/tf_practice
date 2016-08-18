import tensorflow as tf
import numpy as np

def meshgrid(height, width):
    x_t = tf.reshape(tf.matmul(tf.ones([height, 1]),
            tf.expand_dims(tf.linspace(-1., 1., width), 0)), [1, -1])
    y_t = tf.reshape(tf.matmul(tf.expand_dims(tf.linspace(-1., 1., height), 1),
            tf.ones([1, width])), [1, -1])
    return tf.concat(0, [x_t, y_t, tf.ones_like(x_t)])

def repeat(x, num_repeats):
    return tf.reshape(
            tf.matmul(tf.reshape(x, [-1,1]),
                tf.cast(tf.expand_dims(tf.ones(num_repeats), 0), 'int32')),
            [-1])

def interpolate(U, x_s, y_s, out_height, out_width):
    num_batch = tf.shape(U)[0]
    height, width, num_channels = U.get_shape()[1:]

    x_s = 0.5*(x_s + 1.0)*tf.cast(width, 'float32')
    y_s = 0.5*(y_s + 1.0)*tf.cast(height, 'float32')

    x0 = tf.cast(tf.floor(x_s), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y_s), 'int32')
    y1 = y0 + 1
    x0 = tf.clip_by_value(x0, 0, width-1)
    x1 = tf.clip_by_value(x1, 0, width-1)
    y0 = tf.clip_by_value(y0, 0, height-1)
    y1 = tf.clip_by_value(y1, 0, height-1)

    dim1 = height*width
    dim2 = width
    base = repeat(tf.range(num_batch)*dim1, out_height*out_width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2

    U_flat = tf.reshape(U, tf.pack([-1, num_channels]))

    Ia = tf.gather(U_flat, base_y0 + x0)
    Ib = tf.gather(U_flat, base_y1 + x0)
    Ic = tf.gather(U_flat, base_y0 + x1)
    Id = tf.gather(U_flat, base_y1 + x1)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    wa = tf.expand_dims(((x1-x_s)*(y1-y_s)), 1)
    wb = tf.expand_dims(((x1-x_s)*(y_s-y0)), 1)
    wc = tf.expand_dims(((x_s-x0)*(y1-y_s)), 1)
    wd = tf.expand_dims(((x_s-x0)*(y_s-y0)), 1)
    output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return output

def spatial_transformer(U, theta, out_size):
    num_batch = tf.shape(U)[0]
    height, width, num_channels = U.get_shape()[1:]
    out_height, out_width = out_size

    grid = tf.tile(tf.expand_dims(meshgrid(out_height, out_width), 0),
            [num_batch, 1, 1])

    grid_t = tf.batch_matmul(tf.reshape(theta, [-1, 2, 3]), grid)
    x_s = tf.reshape(tf.slice(grid_t, [0,0,0], [-1,1,-1]), [-1])
    y_s = tf.reshape(tf.slice(grid_t, [0,1,0], [-1,1,-1]), [-1])

    return tf.reshape(
            interpolate(U, x_s, y_s, out_height, out_width),
            tf.pack([num_batch, out_height, out_width, num_channels]))

# last layer of localization net
from tensorflow.contrib import layers
def loc_last(input):
    if len(input.get_shape()) == 4:
        input = layers.flatten(input)
    num_inputs = input.get_shape()[1]

    W_init = tf.constant_initializer(np.zeros((num_inputs, 6)))
    b_init = tf.constant_initializer(
            np.array([[1.,0,0],[0,1.,0]]).flatten())
    return layers.fully_connected(input, 6, activation_fn=None,
        weights_initializer=W_init,
        biases_initializer=b_init)

if __name__ == '__main__':
    from scipy import ndimage
    import matplotlib.pyplot as plt
    U = ndimage.imread('gong13.jpg')
    height, width, num_channels = U.shape
    U = U / 255.
    U = U.reshape(1, height, width, num_channels).astype('float32')
    theta = np.array([[1.,0.,0],[0,3.,0]]).reshape(1, 6)

    U_ = tf.placeholder(tf.float32, [None, height, width, num_channels])
    theta_ = tf.placeholder(tf.float32, [None, 6])
    out_size = [5*height, 5*width]
    T_ = spatial_transformer(U_, theta_, out_size)

    with tf.Session() as sess:
        T = sess.run(T_, {U_:U, theta_:theta})
        plt.imshow(T[0])
        plt.show()
