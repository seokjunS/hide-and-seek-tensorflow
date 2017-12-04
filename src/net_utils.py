import tensorflow as tf
import math
import numpy as np
import scipy.misc


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



def multi_crop(data, size=0.75):
  ### do 10 crops
  n, w, h, c = data.shape
  dw = int(w * size)
  dh = int(h * size)

  pw = int((w-dw)/2)
  ph = int((h-dh)/2)


  def five_crop(img):
    idxs = []
    # top left
    idx = np.s_[:, :dw, :dh, :]
    top_left = img[idx]
    idxs.append( idx )

    # top right
    idx = np.s_[:, :dw, (h-dh):, :]
    top_right = img[idx]
    idxs.append( idx )

    # bottom left
    idx = np.s_[:, (w-dw):, :dh, :]
    bottom_left = img[idx]
    idxs.append( idx )

    # bottom right
    idx = np.s_[:, (w-dw):, (h-dh):, :]
    bottom_right = img[idx]
    idxs.append( idx )

    # center
    idx = np.s_[:, pw:(pw+dw), ph:(ph+dh), :]
    center = img[idx]
    idxs.append( idx )

    return [top_left, top_right, bottom_left, bottom_right, center], idxs

  ### calculate
  res, idxs = five_crop(data)
  
  flip_data = data[:, :, ::-1, :] # (n, w, h, c)
  flip_res, flip_idxs = five_crop(flip_data)

  return (res, flip_res), (idxs, flip_idxs), (dw, dh)




def multi_crop_test():
  from scipy import misc
  import matplotlib.pyplot as plt
  from net_utils import multi_crop
  import numpy as np

  img = misc.imread('../temp/images.png', mode='RGB')

  crops, mask, flip_res, flip_mask = multi_crop( np.expand_dims(img, axis=0), size=0.7)

  # [top_left, top_right, bottom_left, bottom_right, center]
  f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2)

  ### ax1: raw image
  ax1.imshow(img[:,::-1,:])
  ax1.set_title('Raw')

  ## ax2: top_left
  ax2.imshow(crops[0][0])
  ax2.set_title('Top Left')

  ## ax3: top_right
  ax3.imshow(crops[1][0])
  ax3.set_title('Top Right')

  ## ax4: bottom_left
  ax4.imshow(crops[2][0])
  ax4.set_title('Bottom Left')

  ## ax5: bottom_right
  ax5.imshow(crops[3][0])
  ax5.set_title('Bottom Right')

  ## ax6: center
  ax6.imshow(crops[4][0])
  ax6.set_title('Center')

  # ## ax7: mask
  # ax7.imshow(mask[0]*20, cmap=plt.cm.jet)
  # ax7.set_title('Mask')

  # ## ax8
  # ax8.imshow(img)
  # ax8.set_title('Raw')

  plt.tight_layout()
  plt.show()

