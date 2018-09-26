import tensorflow as tf
from dependency import *


def weight_variable(shape, name):
    """
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False) # He
    #initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True) # Xaiver 1
    #initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False) # Xaiver 2
    #initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True) # Convolutional
    #initializer = tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='normal')
    return tf.get_variable(initializer=initializer, shape=shape, name=name)


def bias_variable(shape, name):
    """
    """
    #initializer = tf.constant(0.1, shape=shape)
    initializer = tf.ones_initializer()
    return tf.get_variable(initializer=initializer, shape=shape, name=name)


def elu(x):
    with tf.variable_scope("ELU"):
        return tf.nn.elu(x)

def leaky_relu(x, alpha=0.2):
    with tf.variable_scope("LEAKY_RELU"):
        return tf.nn.leaky_relu(x, alpha=alpha)

def param_relu(x, prelu_alpha):
    with tf.variable_scope("PARAM_RELU"):
        return tf.maximum(0.0, x) + prelu_alpha * tf.minimum(0.0, x)

def brelu(x, low_bound=0, up_bound=1):
    with tf.variable_scope("BRELU"):
        return tf.minimum(tf.maximum(low_bound*1.0, x), up_bound*1.0)

def leaky_brelu(x, alpha=0.2, low_bound=0, up_bound=1):
    with tf.variable_scope("LEAKY_BRELU"):
        return tf.minimum(tf.maximum(alpha*x+(1-alpha)*low_bound, x), alpha*x+(1-alpha)*up_bound)


def fully_conn(x, weights, biases):
    return tf.add(tf.matmul(x, weights), biases)


def batch_norm(inputs, training, axis=3):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=axis, momentum=FLAGS.BATCH_MOME, epsilon=FLAGS.BATCH_EPSILON, 
      center=True, scale=True, training=training, fused=True)


def drop_out(x, drop_rate, is_training):
    with tf.variable_scope("DROP_OUT"):
        return tf.layers.dropout(x, drop_rate, training=is_training)


def conv2d(x, filters, biases, strides, padding):
    """
    """
    return tf.nn.conv2d(x, filters, strides=[1, strides[0], strides[1], 1], padding=padding)+ biases


def conv2d_transpose(x, out_dim, filters, biases, strides, padding):
    """
    """
    output_shape = [tf.shape(x)[0]] + out_dim + [filters.get_shape().as_list()[-2]]
    #output_shape[1] *= 2
    #output_shape[2] *= 2
    #import pdb; pdb.set_trace()
    return tf.nn.conv2d_transpose(x, filters, output_shape, strides=[1, strides[0], strides[1], 1], padding=padding) + biases


def max_pool_2x2(x):
    """
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 1, 1, 1], padding='SAME')
