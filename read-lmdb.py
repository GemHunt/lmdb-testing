#from http://deepdish.io/2015/04/28/creating-lmdb-in-python/
#Some example code to read images in LMDB files.

import numpy as np
import lmdb
import sys
sys.path.append('/home/pkrush/caffe/python')
import caffe
import cv2

filename = 'lmdb-test/train_images'
#png: filename = '/home/pkrush/digits/digits/jobs/20160923-121704-e4cc/train_db'
#No encoding:
#filename = '/home/pkrush/digits/digits/jobs/20160923-142347-4f08/train_db'

env = lmdb.open(filename, readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000000')

datum = caffe.proto.caffe_pb2.Datum()

count = 0

with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        datum.ParseFromString(value)
        flat_x = np.fromstring(datum.data, dtype=np.uint8)
        #x = flat_x.reshape(datum.channels, datum.height, datum.width)
        x = flat_x.reshape(datum.height, datum.width)
        y = datum.label
        count += 1
        cv2.imshow('x', x)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break