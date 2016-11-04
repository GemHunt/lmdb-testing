import os
import sys
import time
import glob
import cv2
from random import randint
import random
import matplotlib.pyplot as plt
import infer
import caffe_image
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

def get_whole_rotated_image(crop,mask,angle, crop_size):
    before_rotate_size = 100
    center_x = before_rotate_size/2 + (random.random() * 2) - 1
    center_y = before_rotate_size/2 + (random.random() * 2) - 1

    rot_image = crop.copy()
    rot_image = caffe_image.rotate(rot_image, angle, center_x, center_y, before_rotate_size, before_rotate_size)
    rot_image = cv2.resize(rot_image, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    rot_image = rot_image * mask
    return rot_image

def get_circle_mask(crop_size):
    mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
    cv2.circle(mask, (crop_size/2, crop_size/2), (crop_size/2), 1, cv2.cv.CV_FILLED, lineType=8, shift=0)
    return mask

def infer_one_coin():
    crop_size = 28
    before_rotate_size = 100

    one_coin_rotated = infer.get_classifier("one_coin_rotated", crop_size)
    crop = cv2.imread('/home/pkrush/2-camera-scripts/crops/30287.png')
    if crop is None:
        return

    crop = cv2.resize(crop, (before_rotate_size, before_rotate_size), interpolation=cv2.INTER_AREA)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    mask = get_circle_mask(crop_size)
    diff_angles = []
    scores = []

    for count1 in range(0, 100):
        random_angle = random.random() * 360
        class_angle = int(round(random_angle))
        rot_image = get_whole_rotated_image(crop, mask, random_angle, crop_size)
        rot_image = caffe_image.get_caffe_image(rot_image, crop_size)
        one_coin_rotated_score = one_coin_rotated.predict(rot_image, oversample=False)
        max_value = np.amax(one_coin_rotated_score)
        predicted_angle = np.argmax(one_coin_rotated_score)
        print random_angle, predicted_angle, max_value
        diff_angle = int(round((random_angle - predicted_angle))/5) * 5
        if diff_angle < 0:
            diff_angle += 360
        scores.append(max_value)
        diff_angles.append(diff_angle)


    diff_angles.sort()
    import pandas as pd
    df = pd.DataFrame({'diff':diff_angles,'score':scores})
    print df
    # Create group object
    one = df.groupby('diff')
    print one

    # Apply sum function
    grouped = one.sum()
    print grouped


    grouped.plot()
    plt.show()



    from scipy.interpolate import spline
    xnew = np.linspace(T.min(), T.max(), 300)
    power_smooth = spline(T, power, xnew)

    print scores

    return


def create_lmdbs():
    start_time = time.time()

    max_images = 1000
    crop_size = 28
    before_rotate_size = 100
    classes = 360
    lmdb_dir = '/home/pkrush/lmdb-files'
    shutil.rmtree(lmdb_dir)
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

    for filename in glob.iglob('/home/pkrush/copper/test/*.jpg'):
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

        #print id
        #crop = cv2.imread(filename)


        #crop = cv2.imread('/home/pkrush/copper/test.jpg')
        crop = cv2.imread(filename)
        #if crop is None:
            #continue

        crop = cv2.resize(crop, (before_rotate_size,before_rotate_size), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        mask = get_circle_mask(crop_size)

        for count in range(0, 360):
            angle = float(count)
            class_angle = int(round(angle))

            rot_image = get_whole_rotated_image(crop, mask, angle, crop_size)

            #cv2.imwrite(img_dir + '/' + str(1000 + class_angle) + '/' + str(angle).zfill(5) + '.png',rot_image)

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
    print 'Done after %s seconds' % (time.time() - start_time,)

    return

if __name__ == '__main__':


    create_lmdbs()
    #infer_one_coin()


