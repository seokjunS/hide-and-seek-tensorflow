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







"""
def batch_norm_relu(inputs, is_training):
  ## only work for 1.3
  # inputs = tf.layers.batch_normalization(
  #     inputs=inputs, training=is_training, fused=True)
  inputs = tf.layers.batch_normalization(
      inputs=inputs, training=is_training)
  inputs = tf.nn.relu(inputs)
  return inputs


def conv(inputs, width, in_channels, out_channels, padding='SAME', regularizer=None, bias_init=0.0):
  dtype = inputs.dtype
  # Create variable named "weights".
  weights = tf.get_variable("weights", 
      shape=[1, width, in_channels, out_channels],
      dtype=dtype,
      initializer=tf.contrib.layers.xavier_initializer(),
      regularizer=regularizer)
  # Create variable named "biases".
  biases = tf.get_variable("biases", 
      shape=[out_channels],
      dtype=dtype,
      initializer=tf.constant_initializer(bias_init))
  conv = tf.nn.conv2d(inputs, 
      filter=weights,
      strides=[1, 1, 1, 1], 
      padding=padding)
  return (conv + biases)


def atrous_conv(inputs, width, rate, in_channels, out_channels, padding='SAME', regularizer=None, bias_init=0.0):
  dtype = inputs.dtype
  # Create variable named "weights".
  weights = tf.get_variable("weights", 
      shape=[1, width, in_channels, out_channels],
      dtype=dtype,
      initializer=tf.contrib.layers.xavier_initializer(),
      regularizer=regularizer)
  # Create variable named "biases".
  biases = tf.get_variable("biases", 
      shape=[out_channels],
      dtype=dtype,
      initializer=tf.constant_initializer(bias_init))
  conv = tf.nn.atrous_conv2d(inputs, 
      filters=weights,
      rate=rate, 
      padding=padding)
  return (conv + biases)




def conv_bn_relu(inputs, width, in_channels, out_channels, is_training, padding='SAME', regularizer=None, bias_init=0.0):
  _conv = conv(inputs, width, in_channels, out_channels, padding, regularizer)
  return batch_norm_relu(_conv, is_training)


def atrous_conv_bn_relu(inputs, width, rate, in_channels, out_channels, is_training, padding='SAME', regularizer=None, bias_init=0.0):
  _conv = atrous_conv(inputs, width, rate, in_channels, out_channels, padding, regularizer)
  return batch_norm_relu(_conv, is_training)



def fc(inputs, num_output, activation_fn=tf.nn.relu):
  dtype = inputs.dtype
  i_shape = inputs.get_shape().as_list()

  weights = tf.get_variable("weights",
      shape=[i_shape[-1], num_output],
      initializer=tf.contrib.layers.xavier_initializer())

  biases = tf.get_variable("biases",
      shape=[num_output],
      dtype=dtype,
      initializer=tf.zeros_initializer())

  out = tf.matmul(inputs, weights) + biases
  if activation_fn:
    out = activation_fn( out )

  return out

def embedding(inputs, embedding_size):
  i_shape = inputs.get_shape().as_list()
  embeddings = tf.get_variable("embedding",
      shape=[i_shape[-1], embedding_size],
      dtype=inputs.dtype,
      initializer=tf.contrib.layers.xavier_initializer())
  
  int_inputs = tf.argmax(inputs, axis=2)
  embed = tf.nn.embedding_lookup(embeddings, int_inputs)
  return embed

def highway(inputs, regularizer=None):
  with tf.variable_scope("H"):
    act = conv( inputs,
                width=1,
                in_channels=inputs.get_shape().as_list()[-1], 
                out_channels=inputs.get_shape().as_list()[-1],
                padding='SAME',
                regularizer=regularizer,
                bias_init=-1.0 )
    act = tf.nn.relu( act )
  
  with tf.variable_scope("T"):
    t = conv( inputs,
              width=1,
              in_channels=inputs.get_shape().as_list()[-1], 
              out_channels=inputs.get_shape().as_list()[-1],
              padding='SAME',
              regularizer=regularizer,
              bias_init=0.0 )
    t = tf.nn.sigmoid( t )

  return (t * act) + (1-t) * inputs

def highway_with_H(inputs, H, regularizer=None):  
  with tf.variable_scope("T"):
    t = conv( inputs,
              width=1,
              in_channels=inputs.get_shape().as_list()[-1], 
              out_channels=inputs.get_shape().as_list()[-1],
              padding='SAME',
              regularizer=regularizer,
              bias_init=0.0 )
    t = tf.nn.sigmoid( t )

  return (t * H) + (1-t) * inputs
"""


      