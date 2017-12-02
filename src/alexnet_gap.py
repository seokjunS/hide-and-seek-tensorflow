import tensorflow as tf
from env import *
import net_utils as net



"""
AlexnetGAP model 
"""
"""
Full (simplified) AlexNet architecture:
[227x227x3] INPUT
[55x55x96] CONV1: 96 11x11 filters at stride 4, pad 0
[27x27x96] MAX POOL1: 3x3 filters at stride 2
[27x27x96] NORM1: Normalization layer
[27x27x256] CONV2: 256 5x5 filters at stride 1, pad 2
[13x13x256] MAX POOL2: 3x3 filters at stride 2
[13x13x256] NORM2: Normalization layer
[13x13x384] CONV3: 384 3x3 filters at stride 1, pad 1
[13x13x384] CONV4: 384 3x3 filters at stride 1, pad 1
[13x13x256] CONV5: 256 3x3 filters at stride 1, pad 1
[6x6x256] MAX POOL3: 3x3 filters at stride 2
[4096] FC6: 4096 neurons
[4096] FC7: 4096 neurons
[1000] FC8: 1000 neurons (class scores)
"""

class AlexnetGAP(object):
  def __init__(self,
               num_classes,
               image_mean):
    self.num_classes = num_classes
    self.image_mean = tf.reshape(tf.constant(image_mean), [1,1,1,3])

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

    ### resize image
    x = tf.image.resize_images(self.inputs, size=[RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT])
    # x: [batch, 227, 227, 3]

    ### normalize image
    x = tf.subtract(x, self.image_mean)
    summaries.append( tf.summary.histogram('norm_image', x) )


    ### CONV1
    # [55x55x96] CONV1: 96 11x11 filters at stride 4, pad 0
    # [27x27x96] MAX POOL1: 3x3 filters at stride 2
    with tf.variable_scope('conv_1'):
      x = net.conv_relu_bn(x, 
                           filter_shape=[11,11], 
                           num_filters=96, 
                           stride=4, 
                           padding='VALID', 
                           is_training=self.is_training,
                           regularizer=None)
      # x: [batch, 55, 55, 96]

      x = tf.nn.max_pool(x,
                         ksize=[1,3,3,1],
                         strides=[1,2,2,1],
                         padding='VALID')
      # x: [batch, 27, 27, 96]
      # omit local response normalization. rather using BN

    ### CONV2
    # [27x27x256] CONV2: 256 5x5 filters at stride 1, pad 2
    # [13x13x256] MAX POOL2: 3x3 filters at stride 2
    with tf.variable_scope('conv_2'):
      x = net.conv_relu_bn(x, 
                           filter_shape=[5,5], 
                           num_filters=256, 
                           stride=1, 
                           padding='SAME', 
                           is_training=self.is_training,
                           regularizer=None)
      # x: [batch, 27, 27, 256]

      x = tf.nn.max_pool(x,
                         ksize=[1,3,3,1],
                         strides=[1,2,2,1],
                         padding='VALID')
      # x: [batch, 13, 13, 256]
      # omit local response normalization. rather using BN

    ### CONV3 ~ 5
    # [13x13x384] CONV3: 384 3x3 filters at stride 1, pad 1
    # [13x13x384] CONV4: 384 3x3 filters at stride 1, pad 1
    # [13x13x256] CONV5: 256 3x3 filters at stride 1, pad 1
    with tf.variable_scope('conv_3'):
      x = net.conv_relu_bn(x, 
                           filter_shape=[3,3], 
                           num_filters=384, 
                           stride=1,
                           padding='SAME', 
                           is_training=self.is_training,
                           regularizer=None)
      # x: [batch, 13, 13, 384]
    with tf.variable_scope('conv_4'):
      x = net.conv_relu_bn(x, 
                           filter_shape=[3,3], 
                           num_filters=384, 
                           stride=1,
                           padding='SAME', 
                           is_training=self.is_training,
                           regularizer=None)
      # x: [batch, 13, 13, 384]
    with tf.variable_scope('conv_5'):
      x = net.conv_relu_bn(x, 
                           filter_shape=[3,3], 
                           num_filters=256, 
                           stride=1,
                           padding='SAME', 
                           is_training=self.is_training,
                           regularizer=None)
      # x: [batch, 13, 13, 384]
    
    ### additional conv layer
    with tf.variable_scope('conv_6'):
      x = net.conv_relu_bn(x, 
                           filter_shape=[3,3], 
                           num_filters=512, 
                           stride=1,
                           padding='SAME', 
                           is_training=self.is_training,
                           regularizer=None)
      # x: [batch, 13, 13, 512]
      summaries.append( tf.summary.histogram('conv_6', x) )

    ### GAP
    # [batch, 13, 13, 512] => [batch, 512]
    with tf.variable_scope('gap'):
      # x = tf.reduce_sum(x, axis=[1, 2])
      x = tf.reduce_mean(x, axis=[1, 2])
      # x: [batch, 512]
      summaries.append( tf.summary.histogram('gap', x) )

    ### softmax without bias
    with tf.variable_scope('softmax'):
      num_feature = x.get_shape().as_list()[-1]
      weights = tf.get_variable("weights",
                  shape=[num_feature, self.num_classes],
                  initializer=tf.contrib.layers.xavier_initializer())

      self.logits = tf.matmul(x, weights)
      # logits: [batch, num_classes]

    ### training, loss, ...
    with tf.variable_scope('etc'):
      softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
      softmax_loss = tf.cast(softmax_loss, tf.float64)
      summaries.append(tf.summary.scalar('cross_loss', softmax_loss))

      reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
      reg_loss = tf.cast(reg_loss, tf.float64)
      summaries.append(tf.summary.scalar('reg_loss', reg_loss))

      self.loss_op = softmax_loss + reg_loss
      summaries.append(tf.summary.scalar('total_loss', self.loss_op))

      optimizer = tf.train.MomentumOptimizer( self.learning_rate, 0.9 )
      # optimizer = tf.train.AdamOptimizer( self.learning_rate )
      self.train_op = optimizer.minimize( self.loss_op )

      self.score_op = tf.nn.softmax(self.logits)
      self.pred_op = tf.argmax(self.logits, axis=1)

      self.hit_op = tf.reduce_sum( tf.cast( tf.equal( self.pred_op, self.labels ) , tf.float32) )

      self.summary_op = tf.summary.merge( summaries )

      self.valid_summary_op = tf.summary.scalar('valid_loss', self.valid_loss)



  def train(self, sess, data, labels, learning_rate):
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
    loss, scores, pred = sess.run(
      [self.loss_op, self.score_op, self.pred_op], 
      feed_dict={
        self.inputs: data,
        self.labels: labels,
        self.is_training: False
    })

    return loss, scores, pred


  def inference(self, sess, data):
    pred = sess.run([self.pred_op], feed_dict={
      self.inputs: data,
      self.is_training: False
    })

    return pred


  def summary_valid_loss(self, sess, loss):
    summary = sess.run(self.valid_summary_op, feed_dict={
      self.valid_loss: loss
    })

    return summary