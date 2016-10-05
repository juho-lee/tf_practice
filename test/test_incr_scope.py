import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import variable_scope
#from tensorflow.contrib.layers.python.layers import utils

from utils.nn import fc

x = tf.placeholder(tf.float32, [None, 10])
a = fc(x, 2)
b = fc(a, 2)

def test(input, scope=None, reuse=None):
    with variable_scope.variable_op_scope([input], scope, 'test', reuse=reuse):
        return variables.model_variable('asdf', [1, 1],
                initializer=tf.constant_initializer(0.),
                trainable=True)

c = test(x)
d = test(x)

"""
def get_var(scope=None):
    with variable_scope.variable_scope(scope, 'test'):
        return tf.get_variable('temp',
                [10, 10],
                initializer=tf.constant_initializer(0.0),
                trainable=True)

c = get_var()
"""

vars = tf.trainable_variables()
for var in vars:
    print var.name
