import tensorflow as tf
import numpy as np

class Distribution(object):
    def get_param(self, net_out, **kwargs):
        raise NotImplementedError

    def mean(self, param, **kwargs):
        raise NotImplementedError

    def prior_sample(self, size, **kwargs):
        raise NotImplementedError

    def sample(self, param, **kwargs):
        raise NotImplementedError

    def log_likel(self, x, param, **kwargs):
        raise NotImplementedError

    def kld(self, param, **kwargs):
        raise NotImplementedError

class Bernoulli(Distribution):
    def get_param(self, net_out, **kwargs):
        return {'p': tf.nn.sigmoid(net_out)}

    def mean(self, param, **kwargs):
        return param.get('p')

    def prior_sample(self, size, **kwargs):
        p0 = kwargs.pop('p0', 0.3)
        return np.random.binomial(1, p0, size)
        
        """
        return tf.select(tf.random_uniform(shape) < p0,
                tf.ones(shape), tf.zeros(shape))
        """
        
    def sample(self, param, **kwargs):        
        p = param.get('p')
        shape = tf.shape(p)        
        return tf.select(tf.random_uniform(shape) < p,
                tf.ones(shape), tf.zeros(shape))

    def log_likel(self, x, param, **kwargs):
        p = param.get('p')
        tol = kwargs.pop('tol', 1e-10)
        ll = x*tf.log(p + tol) + (1-x)*tf.log(1-p + tol)
        if len(x.get_shape()) == 4:
            ll = tf.reduce_sum(ll, [1, 2, 3])
        else:
            ll = tf.reduce_sum(ll, 1)
        ll = tf.reduce_mean(ll)
        return ll

class Categorical(Distribution):
    def __init__(self, dim, **kwargs):
        super(Categorical, self).__init__()
        self.dim = dim

    def get_param(self, net_out, **kwargs):
        assert(net_out.get_shape()[1] == self.dim)
        return {'p': tf.nn.softmax(net_out)}

    def mean(self, param, **kwargs):
        return param.get('p')

    def prior_sample(self, size, **kwargs):
        x_int =  np.random.randint(0, self.dim, size).astype(np.int32)
        x = np.zeros((size, self.dim))
        x[range(size), x_int] = 1.
        return x

    def log_likel(self, x, param, **kwargs):
        p = param.get('p')
        tol = kwargs.pop('tol', 1e-10)
        ll = tf.reduce_sum(tf.log(p + tol)*x, 1)
        ll = tf.reduce_mean(ll)
        return ll

from scipy.special import gammaln
class Multinomial(Distribution):
    def __init__(self, dim, ntrials, **kwargs):
        super(Multinomial, self).__init__()
        self.dim = dim
        self.ntrials = ntrials
        #self.log_norm = gammaln(dim+1) - gammaln(ntrials+1) \
        #        - gammaln(dim-ntrials+1)

    def get_param(self, net_out, **kwargs):
        assert(net_out.get_shape()[1] == self.dim)
        return {'p': tf.nn.softmax(net_out)}

    def mean(self, param, **kwargs):
        return param.get('p')

    def prior_sample(self, size, **kwargs):
        x = np.zeros((size, self.dim))
        for i in range(self.ntrials):
            x[range(size), np.random.randint(0, self.dim, size)] += 1.
        return x

    def log_likel(self, x, param, **kwargs):
        p = param.get('p')
        tol = kwargs.pop('tol', 1e-10)
        ll = tf.reduce_sum(tf.log(p + tol)*x, 1)
        ll = tf.reduce_mean(ll)
        return ll

class Gaussian(Distribution):
    def __init__(self, dim, **kwargs):
        super(Gaussian, self).__init__()
        self.dim = dim
        self.fix_var = kwargs.pop('fix_var', False)

    def get_param(self, net_out, **kwargs):
        shape = net_out.get_shape()
        assert(shape.ndims == 2)
        
        if self.fix_var:
            assert(shape[1] == self.dim)
            param = {'mean': net_out}
            return param
        else: 
            assert(shape[1] == 2*self.dim)
            param = {'mean': tf.slice(net_out, [0, 0], [-1, self.dim]), 
                    'log_var': tf.slice(net_out, [0, self.dim], [-1, self.dim])}
            return param

    def mean(self, param, **kwargs):
        return param.get('mean')

    def prior_sample(self, size, **kwargs):
        return np.random.normal(size=(size, self.dim))

    def sample(self, param, **kwargs):        
        mean = param.get('mean')
        log_var = param.get('log_var', tf.zeros_like(mean))
        eps = tf.random_normal(tf.shape(mean))
        return mean + tf.exp(0.5*log_var)*eps

    def log_likel(self, x, param, **kwargs):
        mean = param.get('mean')
        log_var = param.get('log_var', tf.zeros_like(mean))
        c = -0.5*np.log(2*np.pi)
        var = tf.exp(log_var)
        ll = tf.reduce_sum(c - 0.5*log_var - 0.5*tf.square(x - mean) / var, 1)
        ll = tf.reduce_mean(ll)
        return ll

    def kld(self, param, **kwargs):
        mean = param.get('mean')
        log_var = param.get('log_var', tf.zeros_like(mean))
        kld = -0.5*tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), 1)
        kld = tf.reduce_mean(kld)
        return kld

class RectGaussian(Distribution):
    def __init__(self, dim, **kwargs):
        super(RectGaussian, self).__init__()
        self.dim = dim
        self.fix_var = kwargs.pop('fix_var', False)
        self.mean0 = kwargs.pop('mean0', 0.)
        self.log_var0 = kwargs.pop('log_var0', 0.)
        self.std0 = np.exp(0.5*self.log_var0)

    def get_param(self, net_out, **kwargs):
        shape = net_out.get_shape()

        if self.fix_var:
            assert(shape[1] == self.dim)
            param = {'mean': net_out}
            return param
        else: 
            assert(shape[1] == 2*self.dim)
            param = {'mean': tf.slice(net_out, [0, 0], [-1, self.dim]), 
                    'log_var': tf.slice(net_out, [0, dim], [-1, self.dim])}
            return param

    def mean(self, param, **kwargs):
        return param.get('mean')

    def prior_sample(self, size, **kwargs):        
        x = np.random.normal(size=(size, self.dim), 
                loc=self.mean0, scale=self.std0)
        return 0.5*(x + abs(x))

    def sample(self, param, **kwargs):
        mean = param.get('mean')
        log_var = param.get('log_var', tf.zeros_like(mean))
        eps = tf.random_normal(tf.shape(mean))
        return tf.nn.relu(mean + tf.exp(0.5*log_var)*eps)

    def phi(self, x):
        return tf.exp(-0.5*tf.square(x))/np.sqrt(2*np.pi)

    def Phi(self, x):
        return 0.5 + 0.5*tf.erf(x/np.sqrt(2))

    def log_likel(self, x, param, **kwargs):
        mean = param.get('mean')
        log_var = param.get('log_var', tf.zeros_like(mean))
    
        var = tf.exp(log_var)
        log_std = 0.5*log_var
        std = tf.exp(log_std)
        
        p_zero = self.Phi(-mean/std)
        p_nonzero = self.phi((x-mean)/std)/std
        ll = tf.select(tf.equal(x, 0), p_zero, p_nonzero)
        ll = tf.reduce_sum(ll, 1)
        ll = tf.reduce_mean(ll)
        return ll

    def kld(self, param, **kwargs):
        mean = param.get('mean')
        log_var = param.get('log_var', tf.zeros_like(mean))
        mean0 = self.mean0
        log_var0 = self.log_var0
        
        smean = tf.square(mean)
        var = tf.exp(log_var)
        log_std = 0.5*log_var
        std = tf.exp(log_std)

        smean0 = tf.square(mean0)
        var0 = tf.exp(log_var0)
        log_std0 = 0.5*log_var0
        std0 = tf.exp(log_std0)
        
        tol = 1.0e-10
        pzero = self.Phi(-mean/std)
        kld = pzero*(tf.log(pzero+tol) - tf.log(self.Phi(-mean0/std0)+tol))
        kld += (1-pzero)*(log_std0 - log_std + 0.5*(smean0/var0 - smean/var))
        kld += (0.5/var0 - 0.5/var)*((smean + var)*(1-pzero) + mean*std*self.phi(-mean/std))
        kld -= (mean0/var0 - mean/var)*(mean*(1-pzero) + std*self.phi(-mean/std))
        kld = tf.reduce_mean(tf.reduce_sum(kld, 1))
        return kld

from tensorflow.contrib.layers import one_hot_encoding
class MaxOutGaussian(Distribution):
    def __init__(self, **kwargs):
        super(MaxOutGaussian, self).__init__()
        self.fix_var = kwargs.pop('fix_var', True)
        self.mean0 = kwargs.pop('mean0', 0.)

    def get_param(self, net_out, **kwargs):
        shape = net_out.get_shape()
        assert(shape.ndims == 2)                

        if self.fix_var:
            param = {'mean': net_out}
            return param
        else: 
            assert(shape[1] % 2 == 0)
            dim = tf.pack(shape[1] / 2)
            param = {'mean': tf.slice(net_out, [0, 0], [-1, dim]), 
                    'log_var': tf.slice(net_out, [0, dim], [-1, dim])}
            return param

    def mean(self, param, **kwargs):
        return param.get('mean')

    def prior_sample(self, shape, **kwargs):        
        assert(len(shape) == 2)
        tx = np.random.normal(size=shape, loc=self.mean0)
        maxind = np.argmax(tx, axis=1)
        x = np.zeros(shape)
        x[range(shape[0]), maxind] = tx[range(shape[0]), maxind]
        return x

    def sample(self, param, **kwargs):
        mean = param.get('mean')
        log_var = param.get('log_var', tf.zeros_like(mean))
        eps = tf.random_normal(tf.shape(mean))
        tx = mean + tf.exp(0.5*log_var)*eps
        maxval = tf.reduce_max(tx, reduction_indices=1, keep_dims=True)
        maxind = tf.argmax(tx, 1)
        return maxval*one_hot_encoding(maxind, mean.get_shape()[1])

"""
from nn import *
if __name__ == '__main__':    
    x = tf.placeholder(tf.float32, [None, 10])
    q = MaxOutGaussian()
    q_param = q.get_param(linear(x, 5))
    z = q.sample(q_param)
    loss = tf.reduce_mean(tf.reduce_sum(z*z, 1))
    train_op = get_train_op(loss)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    print sess.run(z, {x:np.random.rand(5, 10)})
    sess.run(train_op, {x:np.random.rand(5, 10)})
    print sess.run(z, {x:np.random.rand(5, 10)})
"""


