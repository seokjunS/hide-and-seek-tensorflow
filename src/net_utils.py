import tensorflow as tf
import math


def test_outshape(in_width, filter_width, stride):
  out_width  = math.ceil(float(in_width - filter_width + 1) / float(stride))
  print(out_width)


def relu_bn(inputs, is_training):
  inputs = tf.nn.relu(inputs)
  inputs = tf.layers.batch_normalization(
      inputs=inputs, training=is_training)
  return inputs


def conv(inputs, filter_shape, num_filters, stride, padding, regularizer=None):
  dtype = inputs.dtype
  in_channels = inputs.get_shape().as_list()[-1]
  # Create variable named "weights".
  weights = tf.get_variable("weights", 
      shape=[filter_shape[0], filter_shape[1], in_channels, num_filters],
      dtype=dtype,
      initializer=tf.contrib.layers.xavier_initializer(),
      regularizer=regularizer)
  # Create variable named "biases".
  biases = tf.get_variable("biases", 
      shape=[num_filters],
      dtype=dtype,
      initializer=tf.constant_initializer(0))
  conv = tf.nn.conv2d(inputs, 
      filter=weights,
      strides=[1, stride, stride, 1], 
      padding=padding)
  return (conv + biases)


def conv_relu_bn(inputs, filter_shape, num_filters, stride, padding, is_training, regularizer=None):
  _conv = conv(inputs, filter_shape, num_filters, stride, padding, regularizer)
  return relu_bn(_conv, is_training)


def inception(inputs,
              num_1x1,
              num_3x3_reduce,
              num_3x3,
              num_5x5_reduce,
              num_5x5,
              num_proj,
              is_training,
              regularizer=None):

  with tf.variable_scope('inc_0_1x1'):
    act0 = conv_relu_bn(inputs, 
                        filter_shape=[1,1], 
                        num_filters=num_1x1, 
                        stride=1,
                        padding='SAME', 
                        is_training=is_training,
                        regularizer=regularizer)

  with tf.variable_scope('inc_1_3x3'):
    with tf.variable_scope('reduce'):
      act1 = conv_relu_bn(inputs, 
                          filter_shape=[1,1], 
                          num_filters=num_3x3_reduce, 
                          stride=1,
                          padding='SAME', 
                          is_training=is_training,
                          regularizer=regularizer)
    with tf.variable_scope('conv'):
      act1 = conv_relu_bn(act1, 
                          filter_shape=[3,3], 
                          num_filters=num_3x3, 
                          stride=1,
                          padding='SAME', 
                          is_training=is_training,
                          regularizer=regularizer)

  with tf.variable_scope('inc_2_5x5'):
    with tf.variable_scope('reduce'):
      act2 = conv_relu_bn(inputs, 
                          filter_shape=[1,1], 
                          num_filters=num_5x5_reduce, 
                          stride=1,
                          padding='SAME', 
                          is_training=is_training,
                          regularizer=regularizer)
    with tf.variable_scope('conv'):
      act2 = conv_relu_bn(act2, 
                          filter_shape=[5,5], 
                          num_filters=num_5x5, 
                          stride=1,
                          padding='SAME', 
                          is_training=is_training,
                          regularizer=regularizer)

  with tf.variable_scope('inc_3_proj'):
    with tf.variable_scope('pool'):
      act3 = tf.nn.max_pool(inputs,
                            ksize=[1,3,3,1],
                            strides=[1,1,1,1],
                            padding='SAME')
    with tf.variable_scope('proj'):
      act3 = conv_relu_bn(act3, 
                          filter_shape=[1,1], 
                          num_filters=num_proj, 
                          stride=1,
                          padding='SAME', 
                          is_training=is_training,
                          regularizer=regularizer)

  x = tf.concat(values=[act0, act1, act2, act3], axis=-1)
  return x

