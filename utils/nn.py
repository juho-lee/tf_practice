import tensorflow as tf
fc = tf.contrib.layers.fully_connected
conv = tf.contrib.layers.convolution2d
pool = tf.contrib.layers.max_pool2d
deconv = tf.contrib.layers.convolution2d_transpose
flat = tf.contrib.layers.flatten
dropout = tf.contrib.layers.dropout

def linear(input, num_units, **kwargs):
    return fc(input, num_units, activation_fn=None, **kwargs)

def batch_norm(input, is_train, scope=None, decay=0.9):
    scope = 'BN' if scope is None else scope+'_BN'
    shape = input.get_shape()
    num_out = shape[-1]

    with tf.variable_scope(scope):
        beta = tf.Variable(tf.zeros([num_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.ones([num_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(input, [0,1,2], name='moments') \
                if len(shape)==4 else \
                tf.nn.moments(input, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_train,
                mean_var_with_update,
                lambda: (ema.average(batch_mean), ema.average(batch_var)))
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)

def fc_bn(input, num_units, is_train, scope=None, decay=0.9,
        activation_fn=tf.nn.relu, **kwargs):
    out = linear(input, num_units, scope=scope, **kwargs)
    out = batch_norm(out, is_train, scope=scope, decay=decay)
    out = out if activation_fn is None else activation_fn(out)
    return out

def conv_bn(input, num_ch, filter_size, is_train,
        scope=None, decay=0.9, activation_fn=tf.nn.relu, **kwargs):
    out = conv(input, num_ch, filter_size,
            scope=scope, activation_fn=None, **kwargs)
    out = batch_norm(out, is_train, scope=scope, decay=decay)
    out = out if activation_fn is None else activation_fn(out)
    return out

def deconv_bn(input, num_ch, filter_size, is_train,
        scope=None, decay=0.9, activation_fn=tf.nn.relu, **kwargs):
    out = deconv(input, num_ch, filter_size,
            scope=scope, activation_fn=None, **kwargs)
    out = batch_norm(out, is_train, scope=scope, decay=decay)
    out = out if activation_fn is None else activation_fn(out)
    return out

def get_train_op(loss, var_list=None, learning_rate=None, grad_clip=None):
    if learning_rate is None:
        optimizer = tf.train.AdamOptimizer()
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if grad_clip is None:
        train_op = optimizer.minimize(loss, var_list)
    else:
        gvs = optimizer.compute_gradients(loss, var_list)
        def clip(grad):
            if grad is None:
                return grad
            else:
                return tf.clip_by_value(grad, -grad_clip, grad_clip)
        capped_gvs = [(clip(grad), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
    return train_op
