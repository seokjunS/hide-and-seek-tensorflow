import sys
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import argparse
import time

from env import *
from dataset import Dataset
from alexnet_gap import *
from googlenet_gap import *
# from train import get_model, validation
from train import get_model
from sklearn.metrics import accuracy_score


"""
Set parameters
"""
def arg_parse(args):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--checkpoint',
      type=str,
      default='',
      help='Path of checkpoint file.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size. Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--method',
      type=str,
      default='AlexnetGAP',
      help='Which method?'
  )

  parser.add_argument(
      '--valid_file',
      type=str,
      default=VALID_TFRECORD,
      help='Directory for input data.'
  )


  FLAGS, unparsed = parser.parse_known_args(args)
  print(FLAGS)

  return FLAGS, unparsed




def validation(model, sess, dataset):
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
  print('from hit', hits, hits/num_data)

  return total_loss/num_data, accuracy




def main(sys_argv):
  FLAGS, rest_args = arg_parse(sys_argv)

  
  print("[%s: INFO] Start Evaluation: %s" %
    (datetime.now(), str(FLAGS)))


  with tf.Graph().as_default():
    # valid_set = Dataset(FLAGS.valid_file,
    #                     batch_size=FLAGS.batch_size,
    #                     for_training=False)
    valid_set = Dataset(FLAGS.valid_file,
                          num_data=NUM_CLASSES*NUM_TEST_PER_CLASS,
                          batch_size=FLAGS.batch_size,
                          max_epoch=1,
                          for_training=True)
    # valid_set = Dataset(TRAIN_TFRECORD,
    #                     batch_size=FLAGS.batch_size,
    #                     for_training=False)
    # valid_set = Dataset(TRAIN_TFRECORD,
    #                       num_data=NUM_CLASSES*NUM_TRAIN_PER_CLASS,
    #                       batch_size=FLAGS.batch_size,
    #                       max_epoch=1,
    #                       for_training=True)

    model = get_model(FLAGS)

    with tf.Session() as sess:
      saver = tf.train.Saver()

      # validity of checkpoint
      if not tf.train.checkpoint_exists( FLAGS.checkpoint ):
        print("[%s: ERROR] Checkpoint does not exist! : %s" %
          (datetime.now(), FLAGS.checkpoint))
        return

      saver.restore( sess, FLAGS.checkpoint )

      # loss, accuracy = validation(model, sess, valid_set)
      num_hits = 0.0
      num_data = 0.0
      loss = 0.0
      for data, label, _ in valid_set.iter_batch(sess):
        preds = model.inference(sess, data)
        # print(len(preds), type(preds))
        num_hits += (preds == label).sum()
        num_data += len(preds)

      print(valid_set.num_data, num_data)
      assert(valid_set.num_data == num_data)
      accuracy = num_hits / num_data


      print("[%s: INFO] valuation Result of testset: loss: %.3f, accuracy: %.3f" % 
            (datetime.now(), loss, accuracy))






if __name__ == '__main__':
  args = sys.argv[1:]
  main(args)