from env import *
import os
import glob
import numpy as np
from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt


def gen_labels():
  ### get all human readable dict
  data_path = os.path.join(DATA_PATH, 'tiny-imagenet-200', 'words.txt')

  label2text = dict()

  with open(data_path, 'r') as fr:
    for row in fr.readlines():
      nid, desc = row.rstrip().split('\t')
      label2text[nid] = desc

  ### all directories?
  label_path = os.path.join(DATA_PATH, 'tiny-imagenet-200', 'wnids.txt')

  labels = []

  with open(label_path, 'r') as fr:
    for label in fr.readlines():
      label = label.rstrip()

      labels.append( label )

  ### write labels
  out_path = os.path.join(DATA_PATH, 'labels.txt')
  with open(out_path, 'w') as fw:
    for i, label in enumerate(labels):
      fw.write("%s\t%d\t%s\n" % (label, i, label2text[label]))


"""
return nid->idx dict and idx -> text dict
"""
def load_labels():
  label_path = os.path.join(DATA_PATH, 'labels.txt')

  if not os.path.exists(label_path):
    gen_labels()

  nid2idx = dict()
  idx2text = dict()

  with open(label_path, 'r') as fr:
    for row in fr.readlines():
      nid, idx, text = row.rstrip().split('\t')

      nid2idx[nid] = int(idx)
      idx2text[idx] = text

  return nid2idx, idx2text



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tf_example(image, label, fname, bbox=[0,0,0,0]):
  xmin, ymin, xmax, ymax = bbox

  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image': _bytes_feature(image),
    'label': _int64_feature(label),
    'filename': _bytes_feature(fname.encode('utf8')),
    'xmin': _int64_feature(xmin),
    'ymin': _int64_feature(ymin),
    'xmax': _int64_feature(xmax),
    'ymax': _int64_feature(ymax) 
  }))

  return tf_example

"""
make raw data as TFRecord format.
"""
def gen_data():
  nid2idx, idx2text = load_labels()

  ### make trainset
  print("[INFO] Make trainset")
  writer = tf.python_io.TFRecordWriter(TRAIN_TFRECORD)

  image_path = os.path.join(DATA_PATH, 'tiny-imagenet-200', 'train', '%s', 'images', '*.JPEG')
  annot_path = os.path.join(DATA_PATH, 'tiny-imagenet-200', 'train', '%s', '%s_boxes.txt')

  cnt = 0
  for nid in nid2idx:
    label = nid2idx[nid]
    files = glob.glob( image_path % nid )
    assert(len(files) == NUM_TRAIN_PER_CLASS)

    # get annot
    fname2bbox = dict()
    with open(annot_path%(nid,nid), 'r') as fr:
      for row in fr.readlines():
        fname, xmin, ymin, xmax, ymax = row.rstrip().split("\t")
        fname2bbox[ fname ] = [ int(x) for x in [xmin, ymin, xmax, ymax] ]

    for file in files:
      img = misc.imread(file, mode='RGB')
      fname = file.split("/")[-1]
      bbox = fname2bbox[fname]
      assert(img.shape == (IMAGE_WIDTH, IMAGE_HEIGHT, 3))

      tf_data = create_tf_example(image=img.tobytes(), label=label, fname=fname, bbox=bbox)
      writer.write(tf_data.SerializeToString())

      cnt += 1

  writer.close()


  ### make validation set
  print("[INFO] Make validation set")
  writer = tf.python_io.TFRecordWriter(VALID_TFRECORD)

  # first read nids
  annot_file = os.path.join(DATA_PATH, 'tiny-imagenet-200', 'val', 'val_annotations.txt')

  fname2nid = dict()
  fname2bbox = dict()
  with open(annot_file, 'r') as fr:
    for row in fr.readlines():
      fname, nid, xmin, ymin, xmax, ymax = row.rstrip().split("\t")
      fname2nid[ fname ] = nid
      fname2bbox[ fname ] = [ int(x) for x in [xmin, ymin, xmax, ymax] ]

  # iter images
  image_path = os.path.join(DATA_PATH, 'tiny-imagenet-200', 'val','images', '*.JPEG')

  cnt = 0
  files = glob.glob( image_path )
  assert(len(files) == NUM_TEST_PER_CLASS*NUM_CLASSES)

  for file in files:
    img = misc.imread(file, mode='RGB')
    fname = file.split("/")[-1]
    label = nid2idx[ fname2nid[fname] ]
    bbox = fname2bbox[fname]
    assert(img.shape == (IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    tf_data = create_tf_example(image=img.tobytes(), label=label, fname=fname, bbox=bbox)
    writer.write(tf_data.SerializeToString())

    cnt += 1

  writer.close()


"""
Read TFRecord file into readable format
"""
def read_tfrecord(fname):
  tfrecord_file_queue = tf.train.string_input_producer([fname], name='queue')
  reader = tf.TFRecordReader()
  _, tfrecord_serialized = reader.read(tfrecord_file_queue)

  single_example = tf.parse_single_example(tfrecord_serialized,
                      features={
                          'label': tf.FixedLenFeature([], tf.int64),
                          'image': tf.FixedLenFeature([], tf.string),
                          'filename': tf.FixedLenFeature([], tf.string),
                      }, name='features')
  
  # image was saved as uint8, so we have to decode as uint8.
  image = tf.decode_raw(single_example['image'], tf.uint8)
  image = tf.reshape(image, (IMAGE_WIDTH, IMAGE_HEIGHT, 3))
  label = single_example['label']
  filename = single_example['filename']

  # get real value by feeding
  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    label, image, filename = sess.run([label, image, filename])
    coord.request_stop()
    coord.join(threads)
  
  print(label)
  print(filename)
  plt.imshow(image)
  plt.show() 




def get_valid_image(fname):
  image_path = os.path.join(DATA_PATH, 'tiny-imagenet-200', 'val','images', fname)
  img = misc.imread(image_path, mode='RGB')
  return img





if __name__ == '__main__':
  # gen_data()
  # read_tfrecord(VALID_TFRECORD)
  pass