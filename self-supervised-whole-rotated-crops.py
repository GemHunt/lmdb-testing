# This has Zero Accurarcy when you get away from the 0/360 angle.

# How could this test work better:
# I question if this is mapping images correctly
# Fix:  Count1 is 1-1000 but its only saving 360.
# I bet not doing seq writing to the lmdb and slowing down more.

import os
import sys
import time
import glob
import cv2
from random import randint
import random

import infer
import caffe_image
import caffe_lmdb

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


def infer_two_crops():
    self_super = infer.get_classifier("self-super", 28)

    crop1 = cv2.imread('/home/pkrush/2-camera-scripts/crops/30052.png')
    crop2 = cv2.imread('/home/pkrush/2-camera-scripts/crops/30070.png')
    crops1 = caffe_image.get_angled_crops(crop1, 600)
    crops2 = caffe_image.get_angled_crops(crop2, 600)
    total_max_value = 0
    total_hits = 0
    for angle in range(0, 360):
        combo = np.zeros((28, 28))
        combo[0:14, 0:28] = crops1[angle]
        combo[14:28, 0:28] = crops2[angle]
        self_super_score = self_super.predict(infer.get_caffe_image(combo, 28), oversample=False)
        max_value = np.amax(self_super_score)
        predicted_angle = np.argmax(self_super_score)
        print angle, predicted_angle, max_value
        total_max_value += max_value
        if abs(angle - predicted_angle) < 3:
            total_hits += 1

    print total_max_value, total_hits
    return

def create_lmdbs():
    max_images = 1
    crop_size = 60
    classes = 360
    lmdb_dir = '/home/pkrush/lmdb-files'
    if not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir)
    img_dir = '/home/pkrush/img-files'
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

    for filename in glob.iglob('/home/pkrush/2-camera-scripts/crops/*.png'):
        imageid = filename[-9:]
        imageid = imageid[:5]
        id += 1
        if id > max_images - 1:
            continue

        train_vs_val = randint(1, 4)
        if train_vs_val != 4:
            phase = 'train'
        if train_vs_val == 4:
            phase = 'val'

        print id
        #crop = cv2.imread(filename)
        crop = cv2.imread('/home/pkrush/2-camera-scripts/crops/30070.png')
        if crop is None:
            continue

        crop = cv2.resize(crop, (150,150), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        mask = np.zeros((60,60), dtype=np.uint8)
        cv2.circle(mask, (30, 30), 28, 1, cv2.cv.CV_FILLED, lineType=8, shift=0)


        for count1 in range(0, 10000):
            random_angle = random.random() * 360
            class_angle = int(round(random_angle))
            center_x = 75 + (random.random() * 4) - 2
            center_y = 75 + (random.random() * 4) - 2

            rot_image = crop.copy()
            rot_image = caffe_image.rotate(rot_image, random_angle, center_x, center_y, 150, 150)
            rot_image = cv2.resize(rot_image, (crop_size,crop_size), interpolation=cv2.INTER_AREA)
            rot_image = rot_image * mask
            cv2.imwrite(img_dir + '/' + str(1000 + class_angle) + '/' + imageid + str(count1).zfill(5) + '.png', rot_image)
            rot_image = rot_image.reshape(1, crop_size, crop_size)

            image_sum += rot_image
            str_id = '{:08}'.format(id * 1000 + class_angle)

            # encode into Datum object
            datum = caffe.io.array_to_datum(rot_image, class_angle)
            if phase == 'train':
                train_image_batch.append([str_id, datum])

            if phase == 'val':
                val_image_batch.append([str_id, datum])

        # close databases
        caffe_lmdb._write_batch_to_lmdb(train_image_db, train_image_batch)
        caffe_lmdb._write_batch_to_lmdb(val_image_db, val_image_batch)
        # _write_batch_to_lmdb(label_db, label_batch)
        train_image_batch = []
        val_image_batch = []
        # label_batch = []

    train_image_db.close()
    val_image_db.close()
    # label_db.close()

    # save mean
    mean_image = (image_sum / id * 100).astype('uint8')
    caffe_image.save_mean(mean_image, os.path.join(lmdb_dir, 'mean.binaryproto'))

    return

if __name__ == '__main__':
    start_time = time.time()

    create_lmdbs()

    print 'Done after %s seconds' % (time.time() - start_time,)
