import tensorflow as tf
fc = tf.contrib.layers.fully_connected
conv = tf.contrib.layers.convolution2d
pool = tf.contrib.layers.max_pool2d
deconv = tf.contrib.layers.convolution2d_transpose
flat = tf.contrib.layers.flatten
dropout = tf.contrib.layers.dropout
one_hot = tf.contrib.layers.one_hot_encoding

def linear(inputs, num_outputs, **kwargs):
    return fc(inputs, num_outputs, activation_fn=None, **kwargs)

def batch_norm(inputs, is_training,
        scope=None, reuse=None, decay=0.9):
    shape = inputs.get_shape()
    num_outputs = shape[-1]
    with tf.variable_scope(scope, 'BN', [inputs], reuse=reuse):
        beta = tf.get_variable('beta', [num_outputs],
                initializer=tf.constant_initializer(0.),
                trainable=True)
        gamma = tf.get_variable('gamma', [num_outputs],
                initializer=tf.constant_initializer(1.),
                trainable=True)

        batch_mean, batch_var = tf.nn.moments(inputs, [0,1,2], name='moments') \
                if shape.ndims == 4 else tf.nn.moments(inputs, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training,
                mean_var_with_update,
                lambda: (ema.average(batch_mean), ema.average(batch_var)))
        return tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)

def fc_bn(inputs, num_outputs, is_training, 
        decay=0.9, **kwargs):
    activation_fn = kwargs.pop('activation_fn', tf.nn.relu)
    out = linear(inputs, num_outputs, **kwargs)
    out = batch_norm(out, is_training, 
            scope=kwargs.get('scope'), reuse=kwargs.get('reuse'), decay=decay)
    out = out if activation_fn is None else activation_fn(out)
    return out

def conv_bn(inputs, num_outputs, kernel_size, is_training,
        decay=0.9, **kwargs):
    activation_fn = kwargs.pop('activation_fn', tf.nn.relu)
    out = conv(inputs, num_outputs, kernel_size, activation_fn=None, **kwargs)
    out = batch_norm(out, is_training,
            scope=kwargs.get('scope'), reuse=kwargs.get('reuse'), decay=decay)
    out = out if activation_fn is None else activation_fn(out)
    return out

def deconv_bn(inputs, num_outputs, kernel_size, is_training,
        decay=0.9, **kwargs):
    activation_fn = kwargs.pop('activation_fn', tf.nn.relu)
    out = deconv(inputs, num_outputs, kernel_size, activation_fn=None, **kwargs)
    out = batch_norm(out, is_training,
            scope=kwargs.get('scope'), reuse=kwargs.get('reuse'), decay=decay)
    out = out if activation_fn is None else activation_fn(out)
    return out

def get_train_op(loss,
        var_list=None, grad_clip=None, **kwargs):
    optimizer = tf.train.AdamOptimizer(**kwargs)
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
