'''
Reads images lmdb from databases
Unfinished as a function
Started from http://deepdish.io/2015/04/28/creating-lmdb-in-python/
'''

import numpy as np
import sys

import lmdb

sys.path.append('/home/pkrush/caffe/python')
import caffe
import cv2


def get_batch():
    filename = '/home/pkrush/lmdb-files/val_db'
    filename = '/home/pkrush/digits/digits/jobs/20160923-121704-e4cc/train_db'
    # No encoding:
    # filename = '/home/pkrush/jobs/20161115-123135-f3d7/train_db'
    # filename = '/home/pkrush/lmdb-files/train/1221/train_db'
    filename = '/home/pkrush/lmdb-files/test/0/train_db'

    env = lmdb.open(filename, readonly=True)
    print env.stat()

    # with env.begin() as txn:
    #        raw_datum = txn.get(b'00000000')

    datum = caffe.proto.caffe_pb2.Datum()

    count = 0

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum.ParseFromString(value)
            flat_x = np.fromstring(datum.data, dtype=np.uint8)
            # x = flat_x.reshape(datum.channels, datum.height, datum.width)
            # x = flat_x.reshape(datum.height, datum.width)
            y = datum.label
            print key, y
            count += 1
            # cv2.imshow('x', x)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


get_batch()
