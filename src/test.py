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
from train import get_model, validation

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




def inference(model, sess, dataset):
  dataset.init(sess)

  num_data = dataset.num_data

  labels = np.zeros((num_data))
  preds = np.zeros((num_data))

  sidx = 0
  eidx = 0

  for data, label, _ in dataset.iter_batch(sess):
    pred = model.inference(sess, data)
    samples = data.shape[0]

    sidx = eidx
    eidx += samples
    labels[sidx:eidx] = label
    preds[sidx:eidx] = pred


  accuracy = accuracy_score(labels, preds)
  print(accuracy)

  return accuracy








def main(sys_argv):
  FLAGS, rest_args = arg_parse(sys_argv)

  
  print("[%s: INFO] Start Evaluation: %s" %
    (datetime.now(), str(FLAGS)))


  with tf.Graph().as_default():
    valid_set = Dataset(FLAGS.valid_file,
                        num_data=NUM_CLASSES*NUM_TEST_PER_CLASS,
                        batch_size=FLAGS.batch_size,
                        for_training=False)
    # valid_set = Dataset(TRAIN_TFRECORD,
    #                     num_data=NUM_CLASSES*NUM_TRAIN_PER_CLASS,
    #                     batch_size=FLAGS.batch_size,
    #                     for_training=False)

    model = get_model(FLAGS)

    with tf.Session() as sess:
      saver = tf.train.Saver()

      # validity of checkpoint
      if not tf.train.checkpoint_exists( FLAGS.checkpoint ):
        print("[%s: ERROR] Checkpoint does not exist! : %s" %
          (datetime.now(), FLAGS.checkpoint))
        return

      saver.restore( sess, FLAGS.checkpoint )

      ### test!
      inference(model, sess, valid_set)

      # loss, accuracy = validation(model, sess, valid_set)

      # print("[%s: INFO] valuation Result of testset: loss: %.3f, accuracy: %.3f" % 
      #       (datetime.now(), loss, accuracy))






if __name__ == '__main__':
  args = sys.argv[1:]
  main(args)