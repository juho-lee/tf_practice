import tensorflow as tf
import numpy as np

def gaussian_sample(mean, log_var):
    eps = tf.random_normal(tf.shape(mean))
    return mean + tf.exp(0.5*log_var)*eps

def gaussian_kld(mean, log_var, reduce_mean=True):
    kld = -0.5*tf.reduce_sum(1+log_var-tf.square(mean)-tf.exp(log_var), 1)
    if reduce_mean:
        kld = tf.reduce_mean(kld)
    return kld

# rectified gaussian
def rect_gaussian_sample(mean, log_var):
    eps = tf.random_normal(tf.shape(mean))
    return tf.nn.relu(mean + tf.exp(0.5*log_var)*eps)

def rect_gaussian_kld(mean, log_var, mean0=0., log_var0=0., reduce_mean=True):
    def phi(x):
        return tf.exp(-0.5*tf.square(x))/np.sqrt(2*np.pi)
    def Phi(x):
        return 0.5 + 0.5*tf.erf(x/np.sqrt(2))

    smean = tf.square(mean)
    var = tf.exp(log_var)
    log_std = 0.5*log_var
    std = tf.exp(log_std)

    smean0 = tf.square(mean0)
    var0 = tf.exp(log_var0)
    log_std0 = 0.5*log_var0
    std0 = tf.exp(log_std0)

    tol = 1.0e-10
    pzero = Phi(-mean/std)
    kld = pzero*(tf.log(pzero+tol) - tf.log(Phi(-mean0/std0)+tol))
    kld += (1-pzero)*(log_std0 - log_std + 0.5*(smean0/var0 - smean/var))
    kld += (0.5/var0 - 0.5/var)*((smean + var)*(1-pzero) + mean*std*phi(-mean/std))
    kld -= (mean0/var0 - mean/var)*(mean*(1-pzero) + std*phi(-mean/std))
    kld = tf.reduce_sum(kld, 1)
    if reduce_mean:
        kld = tf.reduce_mean(kld)
    return kld

def bernoulli_neg_ll(x, p, tol=1.0e-10, reduce_mean=True):
    neg_ll = -tf.reduce_sum(x*tf.log(p+tol) + (1.-x)*tf.log(1-p+tol), 1)
    if reduce_mean:
        neg_ll = tf.reduce_mean(neg_ll)
    return neg_ll

def unit_gaussian_neg_ll(x):
    c = -0.5*np.log(2*np.pi)
    neg_ll = tf.reduce_sum(-c + 0.5*tf.square(x), 1)
    return tf.reduce_mean(neg_ll)

def gaussian_neg_ll(x, mean, log_var):
    c = -0.5*np.log(2*np.pi)
    var = tf.exp(log_var)
    neg_ll = tf.reduce_sum(-c + 0.5*log_var + 0.5*tf.square(x - mean) / var, 1)
    return tf.reduce_mean(neg_ll)
