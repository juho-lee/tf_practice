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
    elif theta.get_shape()[1] == 2:
        t_x, t_y = tf.split(1, 2, theta)
        x_s = tf.reshape(tf.tile(x_t, [num_batch,1]) + t_x, [-1])
        y_s = tf.reshape(tf.tile(y_t, [num_batch,1]) + t_y, [-1])
    else:
        grid = tf.expand_dims(tf.concat(0, [x_t, y_t, tf.ones_like(x_t)]), 0)
        grid = tf.tile(grid, [num_batch,1,1])
        grid_t = tf.batch_matmul(tf.reshape(theta, [-1,2,3]), grid)
        x_s = tf.reshape(tf.slice(grid_t, [0,0,0], [-1,1,-1]), [-1])
        y_s = tf.reshape(tf.slice(grid_t, [0,1,0], [-1,1,-1]), [-1])

    return transform(U, x_s, y_s, num_batch, out_height, out_width, num_channels)

def padded_spatial_transformer(U, theta, out_height, out_width):
    height = tf.pack(U.get_shape()[1])
    width = tf.pack(U.get_shape()[2])
    num_channels = tf.pack(U.get_shape()[3])
    
    pad_u = (out_height-height)/2
    pad_d = out_height-height-pad_u
    pad_l = (out_width-width)/2
    pad_r = out_width-width-pad_l

    paddings = [[0, 0], [pad_u, pad_d], [pad_l, pad_r], [0, 0]]
    U_pad = tf.pad(U, paddings)
    U_pad = tf.reshape(U_pad, [-1, out_height, out_width, num_channels])
    return spatial_transformer(U_pad, theta, out_height, out_width)

# last layer of localization net
from tensorflow.contrib import layers
def get_theta(inputs, s_max=None, t_max=None, is_simple=False):
    if len(inputs.get_shape()) == 4:
        inputs = layers.flatten(inputs)
    num_inputs = inputs.get_shape()[1]
    num_outputs = 3 if is_simple else 6

    W_init = tf.constant_initializer(np.zeros((num_inputs, num_outputs)))
    if is_simple:
        b_init = tf.constant_initializer(np.array([1.,0.,0.]))
    else:
        b_init = tf.constant_initializer(np.array([1.,0.,0.,0.,1.,0.]))

    out = layers.fully_connected(inputs, num_outputs,
            activation_fn=None,
            weights_initializer=W_init,
            biases_initializer=b_init)
    if (s_max is not None) | (t_max is not None):
        if is_simple:
            s, tx, ty = tf.split(1, 3, out)
            s = s if s_max is None else tf.clip_by_value(s, -s_max, s_max)
            tx = tx if t_max is None else tf.clip_by_value(tx, -t_max, t_max)
            ty = ty if t_max is None else tf.clip_by_value(ty, -t_max, t_max)
            out = tf.concat(1, [s, tx, ty])
        else:
            sx, vx, tx, vy, sy, ty = tf.split(1, 6, out)
            sx = sx if s_max is None else tf.clip_by_value(sx, -s_max, s_max)
            sy = sy if s_max is None else tf.clip_by_value(sx, -s_max, s_max)
            tx = tx if t_max is None else tf.clip_by_value(tx, -t_max, t_max)
            ty = ty if t_max is None else tf.clip_by_value(ty, -t_max, t_max)
            out = tf.concat(1, [sx, vx, tx, vy, sy, ty])
    return out

def to_trans(inputs, t_max=None):
    if len(inputs.get_shape()) == 4:
        inputs = layers.flatten(inputs)
    num_inputs = inputs.get_shape()[1]
    W_init = tf.constant_initializer(np.zeros((num_inputs, 2)))
    b_init = tf.constant_initializer(np.zeros(2))
    out = layers.fully_connected(inputs, 2,
            activation_fn=None,
            weights_initializer=W_init,
            biases_initializer=b_init)
    if t_max is not None:
        out = t_max*tf.nn.tanh(out)
    return out
    
if __name__ == '__main__':
    from scipy import ndimage
    import matplotlib.pyplot as plt
    U = ndimage.imread('gong13.jpg')
    height, width, num_channels = U.shape
    U = U / 255.
    U = U.reshape(1, height, width, num_channels).astype('float32')

    s = 0.5
    tx = 0.5
    ty = 0.5
    theta = np.array([[s, tx, ty]])
    #s2 = 1.0
    tx2 = 0.0
    ty2 = 0.5
    theta2 = np.array([[tx2, ty2]])
    
    U_ = tf.placeholder(tf.float32, [None, height, width, num_channels])
    theta_ = tf.placeholder(tf.float32, [None, 3])
    theta2_ = tf.placeholder(tf.float32, [None, 2])
    T_ = spatial_transformer(U_, theta_, height/2, width/2)
    T2_ = padded_spatial_transformer(T_, theta2_, height, width)

    with tf.Session() as sess:
        T, T2 = sess.run([T_, T2_], {U_:U, theta_:theta, theta2_:theta2})
        plt.figure()
        plt.imshow(T[0])
        plt.figure()
        plt.imshow(T2[0])
        plt.show()

