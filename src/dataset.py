from env import *
import tensorflow as tf
import numpy as np

"""
Dataset class used for Tensorflow.
Assuming that file is in TFRecord format.
"""
class Dataset:
  def __init__(self, filenames, num_data, batch_size, max_epoch=1, for_training=True):
    self.num_data = num_data
    self.for_training = for_training
    self.dataset = tf.contrib.data.TFRecordDataset(filenames)

    self.dataset = self.dataset.map(self.parser)

    if for_training:
      self.dataset = self.dataset.shuffle(buffer_size=10000)
      self.dataset = self.dataset.batch(batch_size)
      self.dataset = self.dataset.repeat(max_epoch)
      self.iterator = self.dataset.make_one_shot_iterator()
      self.next = self.iterator.get_next()
    else: # reinitializable, iter only one, no shuffle
      self.dataset = self.dataset.batch(batch_size)
      self.iterator = self.dataset.make_initializable_iterator()
      self.next = self.iterator.get_next()
    

  def parser(self, example_proto):
    features = {'label': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string),
                'filename': tf.FixedLenFeature([], tf.string),
                'xmin': tf.FixedLenFeature([], tf.int64),
                'ymin': tf.FixedLenFeature([], tf.int64),
                'xmax': tf.FixedLenFeature([], tf.int64),
                'ymax': tf.FixedLenFeature([], tf.int64) }
    parsed = tf.parse_single_example(example_proto, features)

    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image, (IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    bbox = [parsed['xmin'], parsed['ymin'], parsed['xmax'], parsed['ymax']]

    return image, parsed['label'], parsed['filename'], bbox


  def init(self, sess):
    if self.for_training:
      raise Exception('Training set cannot be reinitialized.')
    else:
      sess.run(self.iterator.initializer)

  
  def iter_batch(self, sess):
    while True:
      try:
        img, label, fname, bbox = sess.run(self.next)
        bbox = np.array(bbox)
        yield img, label, fname, bbox
      except tf.errors.OutOfRangeError:
          break



"""
Aux
calculate mean of all images
"""
def get_stat():
  dataset = Dataset(os.path.join(DATA_PATH, 'train.tfrecord'),
                    batch_size=100,
                    for_training=False)



  with tf.Session() as sess:
    dataset.init(sess)

    s = np.zeros((3))
    cnt = 0.0

    for image, _, _, _ in dataset.iter_batch(sess):
      # image: [batch, w, h, 3]
      n = image.shape[0]
      m = np.mean(image, axis=(1, 2)) # [batch, 3]
      m = np.sum(m, axis=0) # 3

      cnt += n
      s += m

    print( s / cnt )




if __name__ == '__main__':
  # get_stat()




  # dataset = Dataset(os.path.join(DATA_PATH, 'train.tfrecord'),
  #                   batch_size=100,
  #                   max_epoch=10,
  #                   for_training=True)

  cnt = 0
  ### Coord test
  # with tf.Session() as sess:
  #   coord = tf.train.Coordinator()
  #   threads = tf.train.start_queue_runners(coord=coord)
  #   for data in dataset.iter_batch():
  #     try:
  #       a, b, c = sess.run(data)
  #       sess.run(data)
  #       # print(a)
  #       # print(image)
  #       # print('---------')
  #       cnt += 1

  #       if cnt % 50 == 0:
  #         print("%d" % cnt)
  #     except tf.errors.OutOfRangeError:
  #       print("EOS: %d" % cnt)
  #       coord.request_stop()
  #       coord.join(threads)
  #       break
  
  ### without coord test
  # with tf.Session() as sess:
  #   for data in dataset.iter_batch():
  #     try:
  #       a, b, c = sess.run(data)
  #       # print(a)
  #       # print(image)
  #       # print('---------')
  #       cnt += 1

  #       if cnt % 50 == 0:
  #         print("%d" % cnt)
  #     except tf.errors.OutOfRangeError:
  #       print("EOS: %d" % cnt)
  #       break

  ### validation set test
  dataset = Dataset(VALID_TFRECORD,
                    num_data=NUM_CLASSES*NUM_TEST_PER_CLASS,
                    batch_size=100,
                    for_training=False)

  annot_file = os.path.join(DATA_PATH, 'tiny-imagenet-200', 'val', 'val_annotations.txt')
  fname2nid = dict()
  fname2bbox = dict()
  with open(annot_file, 'r') as fr:
    for row in fr.readlines():
      fname, nid, xmin, ymin, xmax, ymax = row.rstrip().split("\t")
      fname2nid[ fname ] = nid
      fname2bbox[ fname ] = [ int(x) for x in [xmin, ymin, xmax, ymax] ]

  cnt = 0
  with tf.Session() as sess:
    dataset.init(sess)

    for data, label, fnames, bboxes in dataset.iter_batch(sess):
      cnt += 1
      for i, fname in enumerate(fnames):
        bbox = bboxes[i]
        for j, a in enumerate(bbox):
          assert( a == fname2bbox[fname][j] )
  print(cnt)

  pass
