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
ALEXNET_IMAGE_WIDTH, ALEXNET_IMAGE_HEIGHT = 227, 227
GOOGLENET_IMAGE_WIDTH, GOOGLENET_IMAGE_HEIGHT = 224, 224

MEAN_IMAGE_RGB = [122.46042058, 114.25709442, 101.36342874]





if __name__ == '__main__':
  if len(sys.argv) != 2 or not sys.argv[1] in globals():
    print("")
  else:
    print( globals()[sys.argv[1]] )
