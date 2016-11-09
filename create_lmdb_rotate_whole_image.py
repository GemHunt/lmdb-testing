import os
import sys
import time
import glob
import cv2
from random import randint
import caffe_image as ci
import caffe_lmdb
import shutil

sys.path.append('/home/pkrush/caffe/python')
sys.path.append('/home/pkrush/digits')

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import lmdb
import numpy as np

if __name__ == '__main__':
    dirname = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(dirname, '..', '..'))
    import digits.config

from digits import utils

# Import digits.config first to set the path to Caffe
import caffe.io
from caffe.proto import caffe_pb2

def create_lmdbs(filenames, create_files = False):
    start_time = time.time()

    max_images = 105
    crop_size = 28
    before_rotate_size = 100
    classes = 360
    lmdb_dir = '/home/pkrush/lmdb-files/train'
    shutil.rmtree(lmdb_dir)
    if not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir)

    if create_files:
        img_dir = '/home/pkrush/img-files'
        shutil.rmtree(img_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for x in range(0, classes):
            img_dir_class = img_dir + '/' + str(1000 + x)
            if not os.path.exists(img_dir_class):
                os.makedirs(img_dir_class)

    # create DBs
    train_image_db = lmdb.open(os.path.join(lmdb_dir, 'train_db'), map_async=True, max_dbs=0)
    val_image_db = lmdb.open(os.path.join(lmdb_dir, 'val_db'), map_async=True, max_dbs=0)

    # add up all images to later create mean image
    image_sum = np.zeros((1, crop_size, crop_size), 'float64')

    # arrays for image and label batch writing
    train_image_batch = []
    val_image_batch = []
    id = -1

    #for filename in glob.iglob('/home/pkrush/copper/test/*.jpg'):
    for filename, angle_offset in filenames:

        #imageid = filename[-9:]
        #imageid = imageid[:5]
        id += 1
        if id > max_images - 1:
            break

        train_vs_val = randint(1, 4)
        if train_vs_val != 4:
            phase = 'train'
        if train_vs_val == 4:
            phase = 'val'

        #crop = cv2.imread('/home/pkrush/copper/test.jpg')
        crop = cv2.imread(filename)
        #if crop is None:
            #continue

        crop = cv2.resize(crop, (before_rotate_size,before_rotate_size), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        mask = ci.get_circle_mask(crop_size)

        for count in range(0, 3600):
            angle = float(count) / 10
            class_angle = int(round(angle))
            angle_to_rotate = angle + angle_offset
            if angle_to_rotate > 360:
                angle_to_rotate - 360


            rot_image = ci.get_whole_rotated_image(crop, mask, angle_to_rotate, crop_size)

            if create_files:
                cv2.imwrite(img_dir + '/' + str(1000 + class_angle) + '/' + str(id) + str(angle).zfill(5) + '.png',rot_image)

            datum = caffe_pb2.Datum()
            datum.data = cv2.imencode('.png', rot_image)[1].tostring()
            #datum.data = cv2.imencode('.png', rot_image).tostring()
            datum.label = class_angle
            datum.encoded = 1

            rot_image = rot_image.reshape(1, crop_size, crop_size)
            image_sum += rot_image
            # datum = caffe.io.array_to_datum(rot_image, class_angle)

            #key = '{:08}'.format(id * 1000 + class_angle)
            #key = '{:08}'.format(angle)

            #if phase == 'train':
            train_image_batch.append([filename + "," + str(angle), datum])

            #if phase == 'val':
                #val_image_batch.append([key, datum])

    caffe_lmdb.write_batch_to_lmdb(train_image_db, train_image_batch)
    caffe_lmdb.write_batch_to_lmdb(val_image_db, val_image_batch)
    # write_batch_to_lmdb(label_db, label_batch)
    train_image_batch = []
    val_image_batch = []
    # label_batch = []

    train_image_db.close()
    val_image_db.close()
    # label_db.close()

    # save mean
    mean_image = (image_sum / id * 100).astype('uint8')
    ci.save_mean(mean_image, os.path.join(lmdb_dir, 'mean.binaryproto'))
    print 'Done after %s seconds' % (time.time() - start_time,)

    return