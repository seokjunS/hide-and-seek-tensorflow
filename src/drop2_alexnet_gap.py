import tensorflow as tf
from env import *
import net_utils as net
import numpy as np



class Drop2AlexnetGAP(object):
  def __init__(self,
               num_classes,
               image_mean,
               do_hide = [],
               without_resize=False):
    self.num_classes = num_classes
    self.l2_reg = 0.001
    self.do_hide = do_hide
    self.without_resize = without_resize
    self.image_mean = np.array(image_mean).reshape((1, 1, 1, 3))
    
    self.tf_image_mean =  tf.constant(image_mean, name='image_mean')

    # placeholders
    with tf.name_scope("Placeholders"):
      # self.inputs = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 3], name='image')
      self.inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='image')
      self.labels = tf.placeholder(tf.int64, shape=[None], name='label')
      self.learning_rate = tf.placeholder(tf.float32, shape=(), name='lr')
      self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
      self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

      self.valid_loss = tf.placeholder(tf.float64, shape=())

    # build
    self.build()
  

  def build(self):
    summaries = []

    ### dropout here
    with tf.variable_scope('dropout'):
      x_shape = tf.shape(self.inputs)
      x_shape = x_shape * tf.constant([1, 1, 1, 0]) + tf.constant([0, 0, 0, 1])
      # batch, w, h, 1
      x = tf.nn.dropout(self.inputs,
                        keep_prob=self.keep_prob,
                        noise_shape=x_shape)

    ### normalize image
    x = tf.subtract(x, self.tf_image_mean)
    # summaries.append( tf.summary.histogram('norm_image', x) )


    ### resize image
    if not self.without_resize:
      x = tf.image.resize_images(x, size=[ALEXNET_IMAGE_WIDTH, ALEXNET_IMAGE_HEIGHT])
      # x: [batch, 227, 227, 3]
    # else:
      # x: [batch, 64, 64, 3]
      # at last conv => 2x2



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
                           regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg))
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
                           regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg))
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
                           regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg))
      # x: [batch, 13, 13, 384]
    with tf.variable_scope('conv_4'):
      x = net.conv_relu_bn(x, 
                           filter_shape=[3,3], 
                           num_filters=384, 
                           stride=1,
                           padding='SAME', 
                           is_training=self.is_training,
                           regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg))
      # x: [batch, 13, 13, 384]
    with tf.variable_scope('conv_5'):
      x = net.conv_relu_bn(x, 
                           filter_shape=[3,3], 
                           num_filters=256, 
                           stride=1,
                           padding='SAME', 
                           is_training=self.is_training,
                           regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg))
      # x: [batch, 13, 13, 384]
    
    ### additional conv layer
    with tf.variable_scope('conv_6'):
      x = net.conv_relu_bn(x, 
                           filter_shape=[3,3], 
                           num_filters=512, 
                           stride=1,
                           padding='SAME', 
                           is_training=self.is_training,
                           regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg))
      self.F = x
      # x: [batch, 13, 13, 512]
      # summaries.append( tf.summary.histogram('conv_6', x) )
      print(self.F)

    ### GAP
    # [batch, 13, 13, 512] => [batch, 512]
    with tf.variable_scope('gap'):
      # x = tf.reduce_sum(x, axis=[1, 2])
      x = tf.reduce_mean(x, axis=[1, 2])
      # x: [batch, 512]
      # summaries.append( tf.summary.histogram('gap', x) )

    ### softmax without bias
    with tf.variable_scope('softmax'):
      num_feature = x.get_shape().as_list()[-1]
      weights = tf.get_variable("weights",
                  shape=[num_feature, self.num_classes],
                  initializer=tf.contrib.layers.xavier_initializer(),
                  regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg))

      self.W = weights
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
    if len(self.do_hide) > 0: # do_hide is num of grid
      N = np.random.choice(self.do_hide, 1)[0]

      ### if N == 0: use full image
      if N != 0:
        n, w, h, _ = data.shape
        mask = net.gen_random_patch(shape=(n, w, h), N=N)
        mask = np.expand_dims(mask, axis=3)

        data = data * mask + (1-mask) * self.image_mean

    _, loss, scores, hits, summary = sess.run(
      [self.train_op, self.loss_op, self.score_op, self.hit_op, self.summary_op],
      feed_dict={
        self.inputs: data,
        self.labels: labels,
        self.learning_rate: learning_rate,
        self.is_training: True,
        self.keep_prob: 0.5
    })

    return loss, scores, hits, summary



  def inference_with_labels(self, sess, data, labels):
    loss, hits, pred = sess.run(
      [self.softmax_loss, self.hit_op, self.pred_op], 
      feed_dict={
        self.inputs: data,
        self.labels: labels,
        self.is_training: False,
        self.keep_prob: 1.0
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
      self.is_training: False,
      self.keep_prob: 1.0
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