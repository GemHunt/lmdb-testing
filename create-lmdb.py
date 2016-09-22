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
sys.path.append('/home/pkrush/digits')

import caffe
id = 0
max_images = 1000
crop_size = 280

env1 = lmdb.open('train-images-db', map_size=int(crop_size*crop_size*3  * max_images * .75 * 1.2))
train_images_txn = env1.begin(write=True)
env2 = lmdb.open('train-labels-db', map_size=10000)
train_labels_txn = env2.begin(write=True)
env3 = lmdb.open('val-images-db', map_size=int((crop_size*crop_size*3) * max_images * .25 * 3.6))
val_images_txn = env3.begin(write=True)
env4 = lmdb.open('val-labels-db', map_size= 10000)
val_labels_txn = env4.begin(write=True)

for filename in glob.iglob('/home/pkrush/2-camera-scripts/crops/*.png'):
    if id > max_images -1:
        break
    print id
    crop = cv2.imread(filename)
    if crop is None:
        continue
    #crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


    for x in range(0,10):
        for y in range(0,10):
            crop_crop = crop[y * 28 :(y + 1) * 28,x * 28 :(x + 1) * 28]
            datum = caffe.io.array_to_datum(crop_crop, -1)
            #datum = caffe.proto.caffe_pb2.Datum()
            #datum.channels = 1
            #datum.height = 28
            #datum.width = 28
            #datum.data = crop_crop.tostring()
            datum.label = x * 10 + y
            str_id = '{:08}'.format(id * 100 + datum.label)
            if (id + 4) & 4 == 0:
                val_images_txn.put(str_id.encode('ascii'), datum.SerializeToString())
            else:
                train_images_txn.put(str_id.encode('ascii'), datum.SerializeToString())
    id += 1

env1.close()
env2.close()
env3.close()
env4.close()


