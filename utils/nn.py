import tensorflow as tf
fc = tf.contrib.layers.fully_connected
conv = tf.contrib.layers.convolution2d
pool = tf.contrib.layers.max_pool2d
deconv = tf.contrib.layers.convolution2d_transpose
flat = tf.contrib.layers.flatten
batch_norm = tf.contrib.layers.batch_norm

def fc_bn(input, num_units, activation_fn=tf.nn.relu, **kwargs):
    output = fc(input, num_units, activation_fn=None, **kwargs)
    return activation_fn(batch_norm(output))

def linear(input, num_units, **kwargs):
    return fc(input, num_units, activation_fn=None, **kwargs)
