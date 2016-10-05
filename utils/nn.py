import tensorflow as tf
fc = tf.contrib.layers.fully_connected
conv = tf.contrib.layers.convolution2d
pool = tf.contrib.layers.max_pool2d
deconv = tf.contrib.layers.convolution2d_transpose
flat = tf.contrib.layers.flatten
dropout = tf.contrib.layers.dropout

def linear(input, num_units, **kwargs):
    return fc(input, num_units, activation_fn=None, **kwargs)

def batch_norm(input, is_train, scope=None, reuse=None, decay=0.9):
    shape = input.get_shape()
    num_out = shape[-1]

    with tf.variable_op_scope([input], scope, 'BN', reuse=reuse):
        beta = tf.get_variable('beta', [num_out],
                initializer=tf.constant_initializer(0.0),
                trainable=True)
        gamma = tf.get_variable('gamma', [num_out],
                initializer=tf.constant_initializer(1.0),
                trainable=True)

        batch_mean, batch_var = tf.nn.moments(input, [0,1,2], name='moments') \
                if len(shape)==4 else tf.nn.moments(input, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_train,
                mean_var_with_update,
                lambda: (ema.average(batch_mean), ema.average(batch_var)))
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)

def fc_bn(input, num_units, is_train,
        scope=None, reuse=None,
        decay=0.9, activation_fn=tf.nn.relu, **kwargs):
    out = linear(input, num_units, **kwargs)
    out = batch_norm(out, is_train, scope=scope, reuse=reuse, decay=decay)
    out = out if activation_fn is None else activation_fn(out)
    return out

def conv_bn(input, num_ch, filter_size, is_train,
        scope=None, reuse=None,
        decay=0.9, activation_fn=tf.nn.relu, **kwargs):
    out = conv(input, num_ch, filter_size, activation_fn=None, **kwargs)
    out = batch_norm(out, is_train, scope=scope, reuse=reuse, decay=decay)
    out = out if activation_fn is None else activation_fn(out)
    return out

def deconv_bn(input, num_ch, filter_size, is_train,
        scope=None, reuse=None,
        decay=0.9, activation_fn=tf.nn.relu, **kwargs):
    out = deconv(input, num_ch, filter_size, activation_fn=None, **kwargs)
    out = batch_norm(out, is_train, scope=scope, reuse=reuse, decay=decay)
    out = out if activation_fn is None else activation_fn(out)
    return out

def get_train_op(loss,
        var_list=None,
        grad_clip=None,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999):

    optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2)
    if grad_clip is None:
        return optimizer.minimize(loss, var_list=var_list)
    else:
        gvs = optimizer.compute_gradients(loss, var_list=var_list)
        def clip(grad):
            if grad is None:
                return grad
            else:
                return tf.clip_by_value(grad, -grad_clip, grad_clip)
        capped_gvs = [(clip(grad), var) for grad, var in gvs]
        return optimizer.apply_gradients(capped_gvs)
