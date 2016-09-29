#Derived from https://github.com/NVIDIA/DIGITS/examples/siamese/create_db.py

import argparse
from collections import defaultdict
import os
import random
import re
import sys
import time
import glob
import cv2
from PIL import Image
from random import randint


sys.path.append('/home/pkrush/caffe/python')
sys.path.append('/home/pkrush/digits')

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import lmdb
import numpy as np
import PIL.Image


if __name__ == '__main__':
    dirname = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(dirname,'..','..'))
    import digits.config

from digits import utils

# Import digits.config first to set the path to Caffe
import caffe.io
from caffe.proto import caffe_pb2


def rotate(img, angle):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    cv2.warpAffine(img, M, (cols, rows),img, cv2.INTER_CUBIC)
    return img;


def create_lmdbs():
    #Creates LMDBs for basic image classification

    max_images = 10000
    crop_size = 60
    folder = 'lmdb-test'

    lmdb_dir = '/home/pkrush/lmdb-files'
    if not os.makedirs(lmdb_dir):
        os.makedirs(lmdb_dir)
    for x in range(0, 360):
        lmdb_dir_class = lmdb_dir + '/' + str(1000 + x)
        if not os.makedirs(lmdb_dir_class):
            os.makedirs(lmdb_dir_class)

    for phase in ('train','val'):
        # create DBs
        image_db = lmdb.open(os.path.join(folder, '%s_db' % phase),
                map_async=True,
                max_dbs=0)
        #label_db = lmdb.open(os.path.join(folder, '%s_labels' % phase),
                #map_async=True,
                #max_dbs=0)

        # add up all images to later create mean image
        image_sum = np.zeros((1,crop_size,crop_size), 'float64')

        # arrays for image and label batch writing
        image_batch = []
        #label_batch = []
        id = -1

        for filename in glob.iglob('/home/pkrush/2-camera-scripts/crops/*.png'):
            imageid = filename[-9:]
            imageid = imageid[:5]
            id += 1
            print id
            if id > max_images - 1:
                continue
            if ((id+4) % 4 == 0) and phase == 'train' :
                continue
            if ((id+4) % 4 != 0) and phase == 'val':
                continue

            crop = cv2.imread(filename)
            if crop is None:
                continue
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(crop, (crop_size * 10, crop_size * 10), interpolation=cv2.INTER_AREA)

            crops = [None] * 360

            #OK, I was lasy here, I really need to remap the crop, not rotate the whole image!
            for angle in range(0,360):
                rotated = rotate(crop.copy(),angle)
                pts1 = np.float32([[440, 295], [580, 290], [440, 305], [580, 310]])
                pts2 = np.float32([[0, 0], [28, 0], [0, 14], [28, 14]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(rotated, M, (28, 14))

                crops[angle] = dst
                # cv2.imshow("warp",rotated)
                # cv2.imshow("dst",dst)
                #if cv2.waitKey(0) & 0xFF == ord('q'):
                    #continue


            for count1 in range(0,1000):
                top = randint(0, 359)
                bottom = top + randint(0, 359)
                if bottom > 359:
                    bottom = bottom - 360

                combo = np.zeros((28,28))
                combo[0:14,0:28] = crops[top]
                combo[14:28,0:28] = crops[bottom]

                cv2.imwrite('lmdb-files/' + str(1000 + top) + '/' + imageid + str(count1 ) + '.png', combo)


                #crop_crop = np.reshape(crop_crop, (1,crop_size,crop_size))
                #image_sum += crop_crop
                #label = x * 10 + y
                #str_id = '{:08}'.format(id * 100 + classid)

                # encode into Datum object
                #datum = caffe.io.array_to_datum(crop_crop, label)
                #image_batch.append([str_id, datum])


        # close databases
        _write_batch_to_lmdb(image_db, image_batch)
        #_write_batch_to_lmdb(label_db, label_batch)
        image_batch = []
        #label_batch = []

        image_db.close()
        #label_db.close()

        if phase == 'train':
            # save mean
            mean_image = (image_sum / id*100).astype('uint8')
            _save_mean(mean_image, os.path.join(folder, 'mean.binaryproto'))

    return

def _write_batch_to_lmdb(db, batch):
    """
    Write a batch of (key,value) to db
    """
    try:
        with db.begin(write=True) as lmdb_txn:
            for key, datum in batch:
                lmdb_txn.put(key, datum.SerializeToString())
    except lmdb.MapFullError:
        # double the map_size
        curr_limit = db.info()['map_size']
        new_limit = curr_limit*2
        try:
            db.set_mapsize(new_limit) # double it
        except AttributeError as e:
            version = tuple(int(x) for x in lmdb.__version__.split('.'))
            if version < (0,87):
                raise Error('py-lmdb is out of date (%s vs 0.87)' % lmdb.__version__)
            else:
                raise e
        # try again
        _write_batch_to_lmdb(db, batch)

def _save_image(image, filename):
    # converting from BGR to RGB
    image = image[[2,1,0],...] # channel swap
    #convert to (height, width, channels)
    image = image.astype('uint8').transpose((1,2,0))
    image = PIL.Image.fromarray(image)
    image.save(filename)

def _save_mean(mean, filename):
    """
    Saves mean to file

    Arguments:
    mean -- the mean as an np.ndarray
    filename -- the location to save the image
    """
    if filename.endswith('.binaryproto'):
        blob = caffe_pb2.BlobProto()
        blob.num = 1
        blob.channels = mean.shape[0]
        blob.height = mean.shape[1]
        blob.width = mean.shape[2]
        blob.data.extend(mean.astype(float).flat)
        with open(filename, 'wb') as outfile:
            outfile.write(blob.SerializeToString())

    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        _save_image(mean, filename)
    else:
        raise ValueError('unrecognized file extension')

if __name__ == '__main__':
    start_time = time.time()

    create_lmdbs()

    print 'Done after %s seconds' % (time.time() - start_time,)

