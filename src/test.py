import sys
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import argparse
import time
import scipy.misc
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches



from env import *
from dataset import Dataset
from alexnet_gap import *
from googlenet_gap import *
from train import get_model, validation
from tiny_imagenet import get_valid_image, load_labels


### GLOBAL
nid2idx, idx2text = load_labels()


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




def inference(model, sess, dataset, localization_thres=0.2, vis_thres=0.9, multi_crop=True, do_vis=False):
  dataset.init(sess)

  num_data = dataset.num_data

  num_label_hits = 0.0
  num_iou_hits = 0.0
  num_all_hits = 0.0
  num_vis = 0

  labels = np.zeros((num_data))
  preds = np.zeros((num_data))
  scores = np.zeros((num_data)) # score of true label
  bboxes = np.zeros((num_data, 4))
  fnames = []

  sidx = 0
  eidx = 0


  for data, label, fname, bbox in dataset.iter_batch(sess):
    pred_label, score, w, f, multi_idxs, multi_sizes = model.inference(sess, data, multi_crop)
    samples = data.shape[0]

    # indexing
    sidx = eidx
    eidx += samples
    
    # extend results
    labels[sidx:eidx] = label
    bboxes[sidx:eidx] = bbox
    preds[sidx:eidx] = pred_label
    label_prob = score[range(samples),label]
    scores[sidx:eidx] = label_prob
    fnames.extend( fname.tolist() )

    # label hit
    label_hit = (label == pred_label).astype(float)
    num_label_hits += label_hit.sum()

    # cam of true label
    cam = make_cam(w, f, label, multi_idxs, multi_sizes)
    pred_bbox = cam_to_bbox(cam, threshold=localization_thres)

    # calc iou
    iou = calc_iou(bbox, pred_bbox)

    # counting hits
    iou_hit = (iou >= 0.5).astype(float)
    num_iou_hits += iou_hit.sum()

    all_hit = label_hit * iou_hit
    num_all_hits += all_hit.sum()

    # visualize
    # vis_idx = (iou >= vis_thres)
    vis_idx = np.logical_and((iou >= vis_thres), (label == pred_label))
    if do_vis:
      visualize_result(sidx=num_vis,
                       cam=cam[vis_idx],
                       iou=iou[vis_idx],
                       true_bbox=bbox[vis_idx],
                       pred_bbox=pred_bbox[vis_idx],
                       label=label[vis_idx],
                       prob=label_prob[vis_idx],
                       fname=fname[vis_idx] )
    num_vis += vis_idx.astype(int).sum()


  top_1_loc = num_all_hits / float(num_data)
  gt_known_loc = num_iou_hits / float(num_data)
  top_1_class = num_label_hits / float(num_data)

  print(top_1_loc, gt_known_loc, top_1_class)
  print('Vis %d'%(int(num_vis)))
  return top_1_class



"""
Args:
  W: (k, c)
  F: (n, w, h, k) or (10*n, w, h, k)
  labels: (n)
  multi_idxs: (list of np.slice, list of np.slice). None if not used
  multi_sizes: (crop width, crop height). None if not used

Returns:
  cam: (n, IMAGE_WIDTH, IMAGE_HEIGHT)
"""
def make_cam(W, F, labels, multi_idxs, multi_sizes):
  # num_n = F.shape[0]
  num_n = labels.shape[0]
  num_k, num_c = W.shape
  
  F = np.expand_dims(F, 3) # (n, w, h, 1, k)
  W = W.reshape((1, 1, 1, num_k, num_c)) # (1, 1, k, c)

  M = np.matmul(F, W) # (n, w, h, 1, c)
  M = np.sum(M, axis=(3)) # (n, w, h, c)

  cam = np.zeros((num_n, IMAGE_WIDTH, IMAGE_HEIGHT))

  # minmax normalization
  max_M = M.max(axis=1).max(axis=1).reshape((-1, 1, 1, num_c)) # (n, 1, 1, c)
  min_M = M.min(axis=1).min(axis=1).reshape((-1, 1, 1, num_c))

  M = (M - min_M) / (max_M - min_M)

  if multi_idxs is None:
    for i in range(num_n):
      cam[i] = scipy.misc.imresize(M[i,:,:,labels[i]], (IMAGE_WIDTH, IMAGE_HEIGHT))
  else:
    # handling multi crop
    (idxs, flip_idxs), (dw, dh) = multi_idxs, multi_sizes
    mask = np.zeros((num_n, IMAGE_WIDTH, IMAGE_HEIGHT))
    cam2 = np.zeros((num_n, IMAGE_WIDTH, IMAGE_HEIGHT))
    mask2 = np.zeros((num_n, IMAGE_WIDTH, IMAGE_HEIGHT))

    for i in range(num_n):
      # for first 5
      for j in range(5):
        M_l = M[(i + num_n * j), :, :, labels[i]]
        xidx, yidx = idxs[j][1], idxs[j][2]

        cam[i, xidx, yidx] += scipy.misc.imresize(M_l, (dw, dh))
        mask[i, xidx, yidx] += 1.0

      # for last 5 -> horizontal flip
      for j in range(5):
        M_l = M[(i + num_n * (j + 5)), :, :, labels[i]]
        xidx, yidx = flip_idxs[j][1], flip_idxs[j][2]

        cam2[i, xidx, yidx] += scipy.misc.imresize(M_l, (dw, dh))
        mask2[i, xidx, yidx] += 1.0

    # concat
    cam += cam2[:, :, ::-1]
    mask += mask2[:, :, ::-1]

    # cam /= mask
    cam /= 10.0

  return cam





"""
Args:
  cam: (n, IMAGE_WIDTH, IMAGE_HEIGHT)

Returns:
  bboxes: (n, 4)

Desc:
  foreground: > 20% of max
"""
def cam_to_bbox(cam, threshold=0.2):
  # get max
  num_n = cam.shape[0]
  bboxes = np.zeros((num_n, 4))
  
  max_v = cam.max(axis=1).max(axis=1).reshape((-1, 1, 1)) # (n, 1, 1)
  thres = max_v * threshold # (n, 1, 1)

  is_foreground = (cam > thres) # (n, w, h)

  binarized = np.zeros(cam.shape, dtype=np.uint8) # (n, w, h)
  binarized[is_foreground] = 1

  for i in range(num_n):
    # find largest connected component
    bimage = binarized[i]
    maxidx, maxval = -1, 0
    components = ndimage.find_objects(bimage)
    for idx, loc in enumerate(components):
      x, y = bimage[loc].shape
      val = x*y

      if val > maxval:
        maxval = val
        maxidx = idx

    loc = components[ maxidx ]
    # xmin, xmax = loc[0].start, loc[0].stop
    # ymin, ymax = loc[1].start, loc[1].stop
    ymin, ymax = loc[0].start, loc[0].stop
    xmin, xmax = loc[1].start, loc[1].stop

    bboxes[i] = [xmin, ymin, xmax, ymax]

  return bboxes



"""
Args:
  bboxes*: (n, 4)

Returns:
  iou: (n)

DESC:
  Calculate IOU
"""
def calc_iou(bboxes1, bboxes2):
  inter = np.stack( [bboxes1, bboxes2], axis=-1 ) # (n, 4, 2)

  maxs = inter[:,:2,:].max(axis=2) # (n, 2) => xmin, ymin
  mins = inter[:,2:,:].min(axis=2) # (n, 2) => xmax, ymax

  inter = np.concatenate( [maxs, mins], axis=1 ) # (n, 4)
  area_inter = area_of_bboxes(inter)

  area1 = area_of_bboxes(bboxes1)
  area2 = area_of_bboxes(bboxes2)

  iou = area_inter.astype(float) / (area1 + area2 - area_inter).astype(float)
  return iou

"""
Args:
  bboxes: (n, 4)

Returns:
  area: (n)
"""
def area_of_bboxes(bboxes):
  area = (bboxes[:,2] - bboxes[:,0] + 1) * (bboxes[:,3] - bboxes[:,1] + 1)
  assert(len(area.shape) == 1 and area.shape[0] == bboxes.shape[0])
  return area





"""
Make image from original image and CAM
"""
def visualize_result(sidx, cam, iou, true_bbox, pred_bbox, label, prob, fname):
  for i in range(len(cam)):
    cam_ = cam[i]
    iou_ = iou[i]
    true_bbox_ = true_bbox[i]
    pred_bbox_ = pred_bbox[i]
    label_ = label[i]
    prob_ = prob[i]
    fname_ = fname[i]
  
    img_ = get_valid_image(fname_)

    # f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    plt.suptitle(fname_)

    ### ax1: raw image
    ax1.imshow(img_)
    ax1.set_title('Raw Image')

    ### ax2: cam
    ax2.imshow(cam_, cmap=plt.cm.jet)
    ax2.set_title('CAM')

    ### ax3: blended
    ax3.imshow(img_)
    ax3.imshow(cam_, cmap=plt.cm.jet, alpha=0.5)
    ax3.set_title('Img+CAM')
    ax3.text(2, 4, 
              "[%d] %s"%(label_, idx2text[str(label_)]), 
              fontweight='bold')
    ax3.text(2, 8, "%.3f"%(prob_), fontweight='bold')

    ### ax4: bbox
    ax4.imshow(img_)
    # true box
    xmin, ymin, xmax, ymax = true_bbox_
    ax4.add_patch(
      patches.Rectangle( (xmin,ymin), (xmax-xmin), (ymax-ymin),
                          fill=False,
                          edgecolor='red',
                          linewidth=4 )
    )
    # pred box
    xmin, ymin, xmax, ymax = pred_bbox_
    ax4.add_patch(
      patches.Rectangle( (xmin,ymin), (xmax-xmin), (ymax-ymin),
                          fill=False,
                          edgecolor='green',
                          linewidth=4 )
    )
    ax4.text(2, 4, "%.3f"%(iou_), fontweight='bold')

    # ### ax5
    # max_v = cam_.max()
    # thres = max_v * 0.2

    # is_foreground = (cam_ > thres)

    # binarized = np.zeros(cam_.shape, dtype=np.uint8)
    # binarized[is_foreground] = 1

    # ax5.imshow(binarized * 100)

    # ### ax6
    # # find largest connected component
    # maxidx, maxval = -1, 0
    # components = ndimage.find_objects(binarized)
    # for idx, loc in enumerate(components):
    #   x, y = binarized[loc].shape
    #   val = x*y

    #   if val > maxval:
    #     maxval = val
    #     maxidx = idx

    # binarized[loc] = 200
    # loc = components[ maxidx ]
    # xmin, xmax = loc[0].start, loc[0].stop
    # ymin, ymax = loc[1].start, loc[1].stop

    # bin_box = [xmin, ymin, xmax, ymax]
    # ax6.imshow(binarized)
    # print(true_bbox_, pred_bbox_, bin_box)


    plt.tight_layout()
    # plt.show()
    plt.savefig('tmp/%s' % (fname_))
    plt.close()    

    # plt.savefig('tmp/%d.png' % i)
    # plt.close()






def main(sys_argv):
  FLAGS, rest_args = arg_parse(sys_argv)

  
  print("[%s: INFO] Start Evaluation: %s" %
    (datetime.now(), str(FLAGS)))


  with tf.Graph().as_default():
    valid_set = Dataset(FLAGS.valid_file,
                        num_data=NUM_CLASSES*NUM_TEST_PER_CLASS,
                        # batch_size=FLAGS.batch_size,
                        batch_size=10,
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
      inference(model, sess, valid_set, localization_thres=0.3, vis_thres=0.9, multi_crop=True, do_vis=True)
      # inference(model, sess, valid_set, localization_thres=0.2, vis_thres=0.9)
      # print("---------------------")
      # inference(model, sess, valid_set, localization_thres=0.3, vis_thres=0.9)
      # print("---------------------")
      # inference(model, sess, valid_set, localization_thres=0.4, vis_thres=0.9)
      # print("---------------------")
      # inference(model, sess, valid_set, localization_thres=0.5, vis_thres=0.9)

      # loss, accuracy = validation(model, sess, valid_set)

      # print("[%s: INFO] valuation Result of testset: loss: %.3f, accuracy: %.3f" % 
      #       (datetime.now(), loss, accuracy))


  print("[%s: INFO] Done" %
    (datetime.now()))



if __name__ == '__main__':
  args = sys.argv[1:]
  main(args)