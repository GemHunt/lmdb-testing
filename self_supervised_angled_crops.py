'''
A sloppy test of questionable value at this point.
Currently this has Zero Accurarcy when you get away from the 0/360 angle.
but, I question if this is mapping images correctly
How could this test work better:
    Fix:  Count1 is 1-1000 but its only saving 360.
    I bet not doing seq writing to the lmdb and slowing down more.
'''

import glob
import os
import sys
import time
from random import randint

import cv2

import caffe_image
import caffe_lmdb
import infer

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

# Import digits.config first to set the path to Caffe
import caffe.io


def infer_two_crops():
    self_super = infer.get_classifier("self-super", 28)

    crop1 = cv2.imread('/home/pkrush/2-camera-scripts/crops/30074.png')
    crop2 = cv2.imread('/home/pkrush/2-camera-scripts/crops/30071.png')
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
    lmdb_dir = '/home/pkrush/train/lmdb-files'
    if not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir)
    img_dir = '/home/pkrush/img-files'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for x in range(0, 360):
        img_dir_class = img_dir + '/' + str(1000 + x)
        if not os.path.exists(img_dir_class):
            os.makedirs(img_dir_class)

    # create DBs
    train_image_db = lmdb.open(os.path.join(lmdb_dir, 'train_db'), map_async=True, max_dbs=0)
    val_image_db = lmdb.open(os.path.join(lmdb_dir, 'val_db'), map_async=True, max_dbs=0)

    # label_db = lmdb.open(os.path.join(folder, '%s_labels' % phase),
    # map_async=True,
    # max_dbs=0)

    # add up all images to later create mean image
    image_sum = np.zeros((1, 28, 28), 'float64')

    # arrays for image and label batch writing
    train_image_batch = []
    val_image_batch = []
    # label_batch = []
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

        crop = cv2.imread(filename)
        if crop is None:
            continue

        crops = caffe_image.get_angled_crops(crop, crop_size * 10)

        for count1 in range(0, 1000):
            # There are 360 classes representing the clockwise travel from
            clockwise_travel_diff_angle = randint(0, 359)
            # The start crop needs to be a random angle because otherwise lens, camera, lighting, compression, cropping, and background effects would falsely trained:
            top = randint(0, 359)
            bottom = top + clockwise_travel_diff_angle
            if bottom > 359:
                bottom -= 360

            combo = np.zeros((28, 28))
            combo[0:14, 0:28] = crops[top]
            combo[14:28, 0:28] = crops[bottom]

            # cv2.imwrite(img_dir + '/' + str(1000 + clockwise_travel_diff_angle) + '/' + imageid + str(count1).zfill(5) + '.png', combo)

            image_sum += combo
            str_id = '{:08}'.format(id * 1000 + clockwise_travel_diff_angle)

            # encode into Datum object
            datum = caffe.io.array_to_datum(combo.reshape(1, 28, 28), clockwise_travel_diff_angle)
            if phase == 'train':
                train_image_batch.append([str_id, datum])

            if phase == 'val':
                val_image_batch.append([str_id, datum])

        # close databases
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
    caffe_image.save_mean(mean_image, os.path.join(lmdb_dir, 'mean.binaryproto'))

    return


if __name__ == '__main__':
    start_time = time.time()

    create_lmdbs()

    print 'Done after %s seconds' % (time.time() - start_time,)
