import tensorflow as tf
fc = tf.contrib.layers.fully_connected
conv = tf.contrib.layers.convolution2d
pool = tf.contrib.layers.max_pool2d
deconv = tf.contrib.layers.convolution2d_transpose
flat = tf.contrib.layers.flatten

def batch_norm(input, is_training, scope, activation_fn=tf.nn.relu):
    return tf.cond(is_training,
        lambda: tf.contrib.layers.batch_norm(input, is_training=True,
            center=True, scale=True,
            updates_collections=None,
            scope=scope, reuse=None,
            activation_fn=activation_fn),
        lambda: tf.contrib.layers.batch_norm(input, is_training=False,
            center=True, scale=True,
            updates_collections=None,
            scope=scope, reuse=True,
            activation_fn=activation_fn))

def linear(input, num_units, **kwargs):
    return fc(input, num_units, activation_fn=None, **kwargs)

def fc_bn(input, num_units, is_training, scope,
        activation_fn=tf.nn.relu, **kwargs):
    out = linear(input, num_units, scope=scope, **kwargs)
    return batch_norm(out, is_training, scope, activation_fn=activation_fn)




