from env import *
import tensorflow as tf


"""
Dataset class used for Tensorflow.
Assuming that file is in TFRecord format.
"""
class Dataset:
  def __init__(self, filenames, batch_size, max_epoch=1, for_training=True):
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
                'filename': tf.FixedLenFeature([], tf.string) }
    parsed = tf.parse_single_example(example_proto, features)

    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image, (IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    return image, parsed['label'], parsed['filename']
    # return parsed['image'], parsed['label'], parsed['filename']


  def init(self, sess):
    if self.for_training:
      raise Exception('Training set cannot be reinitialized.')
    else:
      sess.run(self.iterator.initializer)

  
  def iter_batch(self, sess):
    while True:
      try:
        res = sess.run(self.next)
        yield res
      except tf.errors.OutOfRangeError:
          break







if __name__ == '__main__':
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
  dataset = Dataset(os.path.join(DATA_PATH, 'valid.tfrecord'),
                  batch_size=100,
                  for_training=False)

  with tf.Session() as sess:
    for i in range(10):
      dataset.init(sess)

      for data in dataset.iter_batch(sess):

        a, b, c = data
        # print(a)
        # print(image)
        # print('---------')
        cnt += 1

        if cnt % 50 == 0:
          print("%d" % cnt)

      print("Epoch %d finished" % i, cnt)


  pass
