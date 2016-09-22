#Started from http://deepdish.io/2015/04/28/creating-lmdb-in-python/

import lmdb
import numpy as np
import sys
import glob
import sys
import cv2

# Make sure that caffe is on the python path:
# sys.path.append('~/caffe/python') using the ~ does not work, for some reason???
sys.path.append('/home/pkrush/caffe/python')
import caffe

map_size = 3000000000
env = lmdb.open('mylmdb', map_size=map_size)
id = 0

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for filename in glob.iglob('/home/pkrush/2-camera-scripts/crops/*.png'):
        print filename
        crop = cv2.imread(filename)
        if crop is None:
            continue
        #crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (280, 280), interpolation=cv2.INTER_AREA)

        for x in range(0,9):
            for y in range(0,9):
                crop_crop = crop[y * 28 :(y + 1) * 28,x * 28 :(x + 1) * 28]
                datum = caffe.io.array_to_datum(crop_crop, -1)
                #datum = caffe.proto.caffe_pb2.Datum()
                #datum.channels = 1
                #datum.height = 28
                #datum.width = 28
                #datum.data = crop_crop.tostring()
                datum.label = x * 10 + y
                id += 1
                str_id = '{:08}'.format(id)

                # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
print id