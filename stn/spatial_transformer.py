import tensorflow as tf
import numpy as np

def repeat(x, n):
    return tf.reshape(tf.tile(tf.expand_dims(x, 1), [1,n]), [-1])

def meshgrid(height, width):
    x = tf.tile(tf.linspace(-1.,1.,width), [height])
    y = repeat(tf.linspace(-1.,1.,height), width)
    return x, y

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

def transform(U, x_s, y_s, num_batch, out_height, out_width, num_channels):
    return tf.reshape(
            interpolate(U, x_s, y_s, out_height, out_width),
            tf.pack([num_batch, out_height, out_width, num_channels]))

def spatial_transformer(U, theta, out_height, out_width):
    num_batch = tf.shape(U)[0]
    height, width, num_channels = U.get_shape()[1:]

    x_t, y_t = meshgrid(out_height, out_width)
    x_t = tf.expand_dims(x_t, 0)
    y_t = tf.expand_dims(y_t, 0)
    if theta.get_shape()[1] == 3:
        s, t_x, t_y = tf.split(1, 3, theta)
        x_s = tf.reshape(s*tf.tile(x_t, [num_batch,1]) + t_x, [-1])
        y_s = tf.reshape(s*tf.tile(y_t, [num_batch,1]) + t_y, [-1])
    else:
        grid = tf.expand_dims(tf.concat(0, [x_t, y_t, tf.ones_like(x_t)]), 0)
        grid = tf.tile(grid, [num_batch,1,1])
        grid_t = tf.batch_matmul(tf.reshape(theta, [-1,2,3]), grid)
        x_s = tf.reshape(tf.slice(grid_t, [0,0,0], [-1,1,-1]), [-1])
        y_s = tf.reshape(tf.slice(grid_t, [0,1,0], [-1,1,-1]), [-1])

    return transform(U, x_s, y_s, num_batch, out_height, out_width, num_channels)

# last layer of localization net
from tensorflow.contrib import layers
def to_loc(input, is_simple=False):
    if len(input.get_shape()) == 4:
        input = layers.flatten(input)
    num_inputs = input.get_shape()[1]
    num_outputs = 3 if is_simple else 6
    W_init = tf.constant_initializer(
            np.zeros((num_inputs, num_outputs)))
    if is_simple:
        b_init = tf.constant_initializer(np.array([1.,0.,0.]))
    else:
        b_init = tf.constant_initializer(np.array([1.,0.,0.,0.,1.,0.]))

    return layers.fully_connected(input, num_outputs,
            activation_fn=None,
            weights_initializer=W_init,
            biases_initializer=b_init)

if __name__ == '__main__':
    from scipy import ndimage
    import matplotlib.pyplot as plt
    U = ndimage.imread('gong13.jpg')
    height, width, num_channels = U.shape
    U = U / 255.
    U = U.reshape(1, height, width, num_channels).astype('float32')

    s = -0.5
    t_x = 0.8
    t_y = 0.4
    theta = np.array([[s, 0., t_x, 0., s, t_y]])
    theta_s = np.array([[s, t_x, t_y]])

    U_ = tf.placeholder(tf.float32, [None, height, width, num_channels])
    theta_ = tf.placeholder(tf.float32, [None, 6])
    theta_s_ = tf.placeholder(tf.float32, [None, 3])
    T_ = spatial_transformer(U_, theta_, height, width)
    T_s_ = spatial_transformer(U_, theta_s_, height, width)

    with tf.Session() as sess:
        T, T_s = sess.run([T_, T_s_], {U_:U, theta_:theta, theta_s_:theta_s})
        plt.figure()
        plt.imshow(T[0])
        plt.figure()
        plt.imshow(T_s[0])
        plt.show()
