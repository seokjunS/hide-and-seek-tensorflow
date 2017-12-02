import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_PATH, 'data')

TRAIN_TFRECORD = os.path.join(DATA_PATH, 'train.tfrecord')
VALID_TFRECORD = os.path.join(DATA_PATH, 'valid.tfrecord')



NUM_CLASSES = 200

NUM_TRAIN_PER_CLASS = 500
NUM_TEST_PER_CLASS = 50

IMAGE_WIDTH, IMAGE_HEIGHT = 64, 64
RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT = 227, 227




if __name__ == '__main__':
  if len(sys.argv) != 2 or not sys.argv[1] in globals():
    print("")
  else:
    print( globals()[sys.argv[1]] )
