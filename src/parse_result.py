import os
import glob
from env import *

def parse_results(d, outpath):
  data = dict()
  for file in glob.glob( os.path.join(d, '*.result') ):
    with open(file, 'r') as fr:
      fname = file.strip().split("/")[-1]
      data[ fname ] = dict()
      # skip first line
      fr.readline()
      # iter
      thres = None
      top_1_loc, gt_known_loc, top_1_clas = None, None, None

      for i, row in enumerate(fr.readlines()):
        if i % 3 == 0: # heading line
          thres = row.strip().split()[-1]
        elif i % 3 == 1: # values
          elems = row.strip().replace(',', '').split()
          top_1_loc, gt_known_loc, top_1_clas = elems[2], elems[5], elems[8]
          data[ fname ][ thres ] = (top_1_loc, gt_known_loc, top_1_clas)


  # save 
  thresholds = ['0.20', '0.30', '0.40', '0.50']
  for fname, i in zip( ['top_loc.tsv', 'known_loc.tsv', 'top_clas.tsv'], range(3) ):
    with open(os.path.join(outpath, fname), 'w') as fw:
      # write header
      fw.write("METHOD\t20%\t30%\t40%\t50%\tBest_thres\tBest_val\n")
      # write data
      for method in data:
        if len(data[method]) == 0:
          continue

        maxidx = -1
        maxval = 0.0

        s = "%s\t" % method
        for idx, thres in enumerate(thresholds):
          s += "%s\t" % data[method][thres][i]

          v = float(data[method][thres][i])
          if v > maxval:
            maxval = v
            maxidx = idx

        s += "%s\t%.4f\n" % (thresholds[maxidx], maxval)

        fw.write(s)



if __name__ == '__main__':
  parse_results( os.path.join(BASE_PATH, 'result', 'exp_results'),
                 os.path.join(BASE_PATH, 'result', 'res_tables') )