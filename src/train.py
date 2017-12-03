import sys
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import argparse
import time
import random
from sklearn.metrics import accuracy_score

from env import *
from dataset import Dataset
from alexnet_gap import *
from googlenet_gap import *


"""
Set parameters
"""
def arg_parse(args):
  parser = argparse.ArgumentParser()
  # for learning
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--min_learning_rate',
      type=float,
      default=0.0001,
      help='Min learning rate.'
  )
  parser.add_argument(
      '--max_epoch',
      type=int,
      default=100,
      help='Number of epochs to train.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size. Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--train_file',
      type=str,
      default=TRAIN_TFRECORD,
      help='Directory for input data.'
  )
  parser.add_argument(
      '--valid_file',
      type=str,
      default=VALID_TFRECORD,
      help='Directory for input data.'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='',
      help='Path to write log, checkpoint, and summary files.'
  )
  parser.add_argument(
      '--validation_capacity',
      type=int,
      default=10,
      help='How many validations should be considered for early stopping (--validation_capacity N means considering previous N-1 epochs).'
  )
  parser.add_argument(
      '--log_interval',
      type=int,
      default=100,
      help='Interval of steps for logging.'
  )
  parser.add_argument(
      '--validation_interval',
      type=int,
      default=1,
      help='Interval of steps for logging.'
  )
  parser.add_argument(
      '--method',
      type=str,
      default='GooglenetGAP',
      help='Path to write log, checkpoint, and summary files.'
  )

  FLAGS, unparsed = parser.parse_known_args(args)

  return FLAGS, unparsed





"""
class Monitor
"""
class ValidationMonitor:
  def __init__(self, capacity):
    self.capacity = capacity
    self.slots = [ 100000000 for _ in range(capacity) ]
    self.pointer = 0

  """
  return: should_top, best_index, best_value
  """
  def should_stop(self, new_value):
    # if capacity is 0
    # no early stop
    if self.capacity == 0:
      return False, None, None

    # set new value
    curr_idx = self.pointer % self.capacity
    self.pointer += 1
    self.slots[ curr_idx ] = new_value

    pivot_idx = (curr_idx + 1) % self.capacity # right side
    pivot_value = self.slots[ pivot_idx ]
    # check if all values never imporved
    # in terms of pivot value
    has_drop = False
    for idx, value in enumerate(self.slots):
      if idx != pivot_idx:
        if pivot_value > value:
          has_drop = True
          break

    if has_drop:
      # it dropped, so just keep training
      return False, None, None
    else:
      # no drop, pivot is the best
      return True, max(0, self.pointer - self.capacity), pivot_value


"""
Only save previous one.
if current is same or worse, then return True
"""
class DecayValidationMonitor:
  def __init__(self):
    self.prev = 1000000

  def need_decay(self, new_value):
    if self.prev <= new_value: # loss increases
      self.prev = new_value
      return True
    else:
      self.prev = new_value
      return False




def mkdir(d):
  if not os.path.exists(d):
    os.makedirs(d)


def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, "log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)
  print(msg)



def get_model(FLAGS):
  ### get model
  if FLAGS.method == 'AlexnetGAP':
    model = AlexnetGAP(num_classes=NUM_CLASSES,
                       image_mean=MEAN_IMAGE_RGB)
  elif FLAGS.method == 'GooglenetGAP':
    model = GooglenetGAP(num_classes=NUM_CLASSES,
                         image_mean=MEAN_IMAGE_RGB)
  else:
    model = None

  return model


# def validation(model, sess, dataset):
#   dataset.init(sess)

#   total_loss = 0.0
#   labels = []
#   preds = []
#   scores = []

#   num_data = 0

#   for data, label, _ in dataset.iter_batch(sess):
#     try:
#       loss, score, pred = model.inference_with_labels(sess, data, label)
#     except tf.errors.OutOfRangeError:
#       break

#     samples = data.shape[0]

#     num_data += samples
#     total_loss += loss*samples

#     labels.extend( label.tolist() )
#     preds.extend( pred.tolist() )
#     scores.extend( score.tolist() )
    
#   print(total_loss)
#   print(len(labels))
#   print(labels[:10], pred[:10])
#   accuracy = accuracy_score(labels, preds)

#   return total_loss/num_data, accuracy



def validation(model, sess, dataset):
  dataset.init(sess)

  num_data = dataset.num_data
  total_loss = 0.0
  hits = 0.0

  labels = np.zeros((num_data))
  preds = np.zeros((num_data))

  sidx = 0
  eidx = 0

  for data, label, _ in dataset.iter_batch(sess):
    loss, hit, pred = model.inference_with_labels(sess, data, label)
    samples = data.shape[0]

    total_loss += loss*samples
    hits += hit

    sidx = eidx
    eidx += samples
    labels[sidx:eidx] = label
    preds[sidx:eidx] = pred


  accuracy = accuracy_score(labels, preds)
  # print('from hit', hits, hits/num_data)

  return total_loss/num_data, accuracy




def main(sys_argv):
  FLAGS, rest_args = arg_parse(sys_argv)
  
  ### prepare directories
  mkdir(FLAGS.train_dir)
  FLAGS.checkpoint_path = os.path.join(FLAGS.train_dir, 'ckpt', 'model')
  mkdir(os.path.join(FLAGS.train_dir, 'ckpt'))
  FLAGS.log_dir = os.path.join(FLAGS.train_dir)
  FLAGS.summary_dir = os.path.join(FLAGS.train_dir, 'summary')
  mkdir(FLAGS.summary_dir)

  logging("[%s: INFO] Setup: %s" % 
              (datetime.now(), str(FLAGS)), FLAGS)
  


  with tf.Graph().as_default():
    ### get dataset
    train_set = Dataset(FLAGS.train_file,
                          num_data=NUM_CLASSES*NUM_TRAIN_PER_CLASS,
                          batch_size=FLAGS.batch_size,
                          max_epoch=FLAGS.max_epoch,
                          for_training=True)

    valid_set = Dataset(FLAGS.valid_file,
                        num_data=NUM_CLASSES*NUM_TEST_PER_CLASS,
                        batch_size=FLAGS.batch_size,
                        for_training=False)

    #TODO
    model = get_model(FLAGS)
    # monitor = ValidationMonitor(capacity=FLAGS.validation_capacity)
    monitor = DecayValidationMonitor()


    with tf.Session() as sess:
      learning_rate = FLAGS.learning_rate

      saver = tf.train.Saver(max_to_keep=FLAGS.validation_capacity)
      writer = tf.summary.FileWriter(FLAGS.summary_dir)

      sess.run( tf.global_variables_initializer() )

      cnt_epoch = 0
      for step, (data, labels, _) in enumerate(train_set.iter_batch(sess)):
        start_time = time.time()
        loss, scores, hits, summary = model.train(sess, data, labels, learning_rate)
        duration = time.time() - start_time


        if step % FLAGS.log_interval == 0:
          examples_per_sec = FLAGS.batch_size / float(duration)

          logging("[%s: INFO] %d step => loss: %.3f, acc: %.3f (%.1f examples/sec; %.3f sec/batch)" % 
            (datetime.now(), step, loss, hits/FLAGS.batch_size, examples_per_sec, duration), FLAGS)

          writer.add_summary(summary, step)
          # print('pred', pred)
          # print('labels', labels)

        ### validation
        # if (step+1) % int(NUM_CLASSES * NUM_TRAIN_PER_CLASS / FLAGS.batch_size) == 0:
        if (step+1) % 10 == 0:
          cnt_epoch += 1
          logging("[%s: INFO] %d epoch done!" % 
              (datetime.now(), cnt_epoch), FLAGS)

          saver.save(sess, FLAGS.checkpoint_path, global_step=cnt_epoch)
          
          loss, accuracy = validation(model, sess, valid_set)

          logging("[%s: INFO] Validation Result at %d epochs: loss: %.3f, accuracy: %.3f" % 
            (datetime.now(), cnt_epoch, loss, accuracy), FLAGS)

          valid_summary = model.summary_valid_loss(sess, loss)
          writer.add_summary(valid_summary, step)

          ### rate decaying check
          # gradually decaying
          new_lr = (FLAGS.learning_rate - FLAGS.min_learning_rate) * (1 - cnt_epoch/(FLAGS.max_epoch*1.0)) + FLAGS.min_learning_rate
          logging("[%s: INFO] LR decay at %d. %f =>  %f" % 
                (datetime.now(), cnt_epoch, learning_rate, new_lr), FLAGS)
          learning_rate = new_lr

          # need_decay = monitor.need_decay(loss)

          # if need_decay:
          #   new_lr = learning_rate * 0.1

          #   if new_lr < FLAGS.min_learning_rate:
          #     logging("[%s: INFO] STOP! at %d. LR: %.f" % 
          #       (datetime.now(), cnt_epoch, learning_rate), FLAGS)
          #     break
          #   else:
          #     logging("[%s: INFO] LR decay at %d. %.f =>  %.f" % 
          #       (datetime.now(), cnt_epoch, learning_rate, new_lr), FLAGS)
          #     learning_rate = new_lr
            







if __name__ == '__main__':
  args = sys.argv[1:]
  main(args)
