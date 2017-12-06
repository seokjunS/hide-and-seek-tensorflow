import tensorflow as tf
from env import *
import net_utils as net
import numpy as np


"""
GooglenetGAP model 
"""

class GooglenetGAP(object):
  def __init__(self,
               num_classes,
               image_mean,
               do_hide = None):
    self.num_classes = num_classes
    self.l2_reg = 0.001
    self.do_hide = do_hide
    self.image_mean = np.array(image_mean).reshape((1, 1, 1, 3))

    self.tf_image_mean =  tf.constant(image_mean, name='image_mean')

    # placeholders
    with tf.name_scope("Placeholders"):
      self.inputs = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 3], name='image')
      self.labels = tf.placeholder(tf.int64, shape=[None], name='label')
      self.learning_rate = tf.placeholder(tf.float32, shape=(), name='lr')
      self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

      self.valid_loss = tf.placeholder(tf.float64, shape=())

    # build
    self.build()
  

  def build(self):
    summaries = []

    ### normalize image
    x = tf.subtract(self.inputs, self.tf_image_mean)
    # summaries.append( tf.summary.histogram('norm_image', x) )

    ### resize image
    x = tf.image.resize_images(x, size=[GOOGLENET_IMAGE_WIDTH, GOOGLENET_IMAGE_HEIGHT])
    # x: [batch, 224, 224, 3]


    ### CONV1: 7x7 / 2 [64]
    with tf.variable_scope('conv_1'):
      x = net.conv_relu_bn(x, 
                           filter_shape=[7,7], 
                           num_filters=64, 
                           stride=2, 
                           padding='SAME', 
                           is_training=self.is_training,
                           regularizer=None)
      # x: [batch, 112, 112, 64]

      x = tf.nn.max_pool(x,
                         ksize=[1,3,3,1],
                         strides=[1,2,2,1],
                         padding='SAME')
      # x: [batch, 56, 56, 64]

    ### CONV2: (3x3_reduce / 1 [64]) + (3x3 / 1 [192])
    with tf.variable_scope('conv_2'):
      with tf.variable_scope('reduce'):
        x = net.conv_relu_bn(x, 
                             filter_shape=[1,1], 
                             num_filters=64, 
                             stride=1, 
                             padding='SAME', 
                             is_training=self.is_training,
                             regularizer=None)
        # x: [batch, 56, 56, 64]

      with tf.variable_scope('conv'):
        x = net.conv_relu_bn(x, 
                             filter_shape=[3,3], 
                             num_filters=192, 
                             stride=1, 
                             padding='SAME', 
                             is_training=self.is_training,
                             regularizer=None)
        # x: [batch, 56, 56, 192]

      x = tf.nn.max_pool(x,
                         ksize=[1,3,3,1],
                         strides=[1,2,2,1],
                         padding='SAME')
      # x: [batch, 28, 28, 192]


    ### INCEPTION (3a)
    with tf.variable_scope('inception_3a'):
      x = net.inception(x,
                        num_1x1=64,
                        num_3x3_reduce=96,
                        num_3x3=128,
                        num_5x5_reduce=16,
                        num_5x5=32,
                        num_proj=32,
                        is_training=self.is_training,
                        regularizer=None)
      # x: [batch, 28, 28, 256]

    ### INCEPTION (3b)
    with tf.variable_scope('inception_3b'):
      x = net.inception(x,
                        num_1x1=128,
                        num_3x3_reduce=128,
                        num_3x3=192,
                        num_5x5_reduce=32,
                        num_5x5=96,
                        num_proj=64,
                        is_training=self.is_training,
                        regularizer=None)
      # x: [batch, 28, 28, 480]

      x = tf.nn.max_pool(x,
                         ksize=[1,3,3,1],
                         strides=[1,2,2,1],
                         padding='SAME')
      # x: [batch, 14, 14, 480]

    ### INCEPTION (4a)
    with tf.variable_scope('inception_4a'):
      x = net.inception(x,
                        num_1x1=192,
                        num_3x3_reduce=96,
                        num_3x3=208,
                        num_5x5_reduce=16,
                        num_5x5=48,
                        num_proj=64,
                        is_training=self.is_training,
                        regularizer=None)
      # x: [batch, 14, 14, 512]

    ### INCEPTION (4b)
    with tf.variable_scope('inception_4b'):
      x = net.inception(x,
                        num_1x1=160,
                        num_3x3_reduce=112,
                        num_3x3=224,
                        num_5x5_reduce=24,
                        num_5x5=64,
                        num_proj=64,
                        is_training=self.is_training,
                        regularizer=None)
      # x: [batch, 14, 14, 512]
    
    ### INCEPTION (4c)
    with tf.variable_scope('inception_4c'):
      x = net.inception(x,
                        num_1x1=128,
                        num_3x3_reduce=128,
                        num_3x3=256,
                        num_5x5_reduce=24,
                        num_5x5=64,
                        num_proj=64,
                        is_training=self.is_training,
                        regularizer=None)
      # x: [batch, 14, 14, 512]

    ### INCEPTION (4d)
    with tf.variable_scope('inception_4d'):
      x = net.inception(x,
                        num_1x1=112,
                        num_3x3_reduce=144,
                        num_3x3=288,
                        num_5x5_reduce=32,
                        num_5x5=64,
                        num_proj=64,
                        is_training=self.is_training,
                        regularizer=None)
      # x: [batch, 14, 14, 528]

    ### INCEPTION (4e)
    with tf.variable_scope('inception_4e'):
      x = net.inception(x,
                        num_1x1=256,
                        num_3x3_reduce=160,
                        num_3x3=320,
                        num_5x5_reduce=32,
                        num_5x5=128,
                        num_proj=128,
                        is_training=self.is_training,
                        regularizer=None)
      # x: [batch, 14, 14, 832]

    
    ### additional conv layer
    with tf.variable_scope('conv_5'):
      x = net.conv_relu_bn(x, 
                           filter_shape=[3,3], 
                           num_filters=1024, 
                           stride=1,
                           padding='SAME', 
                           is_training=self.is_training,
                           regularizer=None)
      self.F = x
      # x: [batch, 14, 14, 1024]
      # summaries.append( tf.summary.histogram('conv_5', x) )

    ### GAP
    # [batch, 14, 14, 512] => [batch, 1024]
    with tf.variable_scope('gap'):
      # x = tf.reduce_sum(x, axis=[1, 2])
      x = tf.reduce_mean(x, axis=[1, 2])
      # x: [batch, 1024]
      # summaries.append( tf.summary.histogram('gap', x) )

    ### softmax without bias
    with tf.variable_scope('softmax'):
      num_feature = x.get_shape().as_list()[-1]
      weights = tf.get_variable("weights",
                  shape=[num_feature, self.num_classes],
                  initializer=tf.contrib.layers.xavier_initializer(),
                  regularizer=None)

      self.logits = tf.matmul(x, weights)
      # logits: [batch, num_classes]

    ### training, loss, ...
    with tf.variable_scope('etc'):
      self.softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
      self.softmax_loss = tf.cast(self.softmax_loss, tf.float64)
      summaries.append(tf.summary.scalar('cross_loss', self.softmax_loss))

      reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
      reg_loss = tf.cast(reg_loss, tf.float64)
      summaries.append(tf.summary.scalar('reg_loss', reg_loss))

      self.loss_op = self.softmax_loss + reg_loss
      summaries.append(tf.summary.scalar('total_loss', self.loss_op))

      optimizer = tf.train.MomentumOptimizer( self.learning_rate, 0.9 )
      # optimizer = tf.train.AdamOptimizer( self.learning_rate )
      
      updates_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(updates_ops):
        self.train_op = optimizer.minimize( self.loss_op )

      self.score_op = tf.nn.softmax(self.logits)
      
      self.pred_op = tf.argmax(self.logits, axis=1)

      self.hit_op = tf.reduce_sum( tf.cast( tf.equal( self.pred_op, self.labels ) , tf.float32) )

      self.summary_op = tf.summary.merge( summaries )

      self.valid_summary_op = tf.summary.scalar('valid_loss', self.valid_loss)



  def train(self, sess, data, labels, learning_rate):
    ### do hiding?
    if self.do_hide is not None: # do_hide is num of grid
      n, w, h, _ = data.shape
      mask = net.gen_random_patch(shape=(n, w, h), N=self.do_hide)
      mask = np.expand_dims(mask, axis=3)

      data = data * mask + (1-mask) * self.image_mean

    _, loss, scores, hits, summary = sess.run(
      [self.train_op, self.loss_op, self.score_op, self.hit_op, self.summary_op],
      feed_dict={
        self.inputs: data,
        self.labels: labels,
        self.learning_rate: learning_rate,
        self.is_training: True
    })

    return loss, scores, hits, summary



  def inference_with_labels(self, sess, data, labels):
    loss, hits, pred = sess.run(
      [self.softmax_loss, self.hit_op, self.pred_op], 
      feed_dict={
        self.inputs: data,
        self.labels: labels,
        self.is_training: False
    })

    return loss, hits, pred


  def inference(self, sess, data, multi_crop=False):
    if multi_crop:
      raw_data = data
      (crops, flip_crops), idxs, sizes = net.multi_crop(data, size=0.75)
      data = np.concatenate( crops + flip_crops, axis=0 )
      # print(raw_data.shape, data.shape)
    else:
      idxs = sizes = None

    pred, score, W, F = sess.run([self.pred_op, self.score_op, self.W, self.F], feed_dict={
      self.inputs: data,
      self.is_training: False
    })

    if multi_crop:
      # print(score.shape)
      num_n, num_classes = score.shape[0]/10, score.shape[1]
      shape = (10, num_n, num_classes)
      score = score.reshape(shape)
      score = np.mean(score, axis=0)

      pred = np.argmax(score, axis=1)


    return pred, score, W, F, idxs, sizes


  def summary_valid_loss(self, sess, loss):
    summary = sess.run(self.valid_summary_op, feed_dict={
      self.valid_loss: loss
    })

    return summary