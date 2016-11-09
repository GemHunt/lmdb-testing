'''
This makes crops up images on a 10x10 grid and creates lmdbs will 100 classes
It's interesting that this works somewhat in part because of lens, lighting, camera, etc effects.
'''

import os
import sys
import time
import glob
import cv2
import lmdb
import numpy as np
import caffe_image
import caffe_lmdb

sys.path.append('/home/pkrush/caffe/python')
sys.path.append('/home/pkrush/digits')

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO


if __name__ == '__main__':
    dirname = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(dirname,'..','..'))
    import digits.config

from digits import utils

# Import digits.config first to set the path to Caffe
import caffe.io
from caffe.proto import caffe_pb2


def create_lmdbs():
    #Creates LMDBs for basic image classification

    max_images = 10000
    crop_size = 280
    folder = 'lmdb-test'

    for phase in ('train','val'):
        # create DBs
        image_db = lmdb.open(os.path.join(folder, '%s_db' % phase),
                map_async=True,
                max_dbs=0)
        #label_db = lmdb.open(os.path.join(folder, '%s_labels' % phase),
                #map_async=True,
                #max_dbs=0)

        # add up all images to later create mean image
        image_sum = np.zeros((1,28,28), 'float64')

        # arrays for image and label batch writing
        image_batch = []
        #label_batch = []
        id = -1

        for filename in glob.iglob('/home/pkrush/2-camera-scripts/crops/*.png'):
            imageid = filename[-9:]
            imageid = imageid[:5]
            id += 1
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
            crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

            #os.makedirs('lmdb-files')
            #for x in range(0, 10):
                #for y in range(0, 10):
                    #os.makedirs('lmdb-files/' + str(x * 10 + y + 100))

            for x in range(0, 10):
                for y in range(0, 10):
                    crop_crop = crop[x * 28:(x + 1) * 28, y * 28:(y + 1) * 28]
                    classid = x * 10 + y + 100
                    #cv2.imwrite('lmdb-files/' + str(classid) + '/' + imageid + '.png', crop_crop)
                    crop_crop = np.reshape(crop_crop, (1,28,28))
                    image_sum += crop_crop
                    label = x * 10 + y
                    str_id = '{:08}'.format(id * 100 + classid)

                    # encode into Datum object
                    datum = caffe.io.array_to_datum(crop_crop, label)
                    image_batch.append([str_id, datum])

                    # create label Datum
                    #label_datum = caffe_pb2.Datum()
                    #label_datum.channels, label_datum.height, label_datum.width = 1, 1, 1
                    #label_datum.float_data.extend(np.array([label]).flat)
                    #label_batch.append([str_id, label_datum])


        # close databases
        caffe_lmdb.write_batch_to_lmdb(image_db, image_batch)
        #write_batch_to_lmdb(label_db, label_batch)
        image_batch = []
        #label_batch = []

        image_db.close()
        #label_db.close()

        if phase == 'train':
            # save mean
            mean_image = (image_sum / id*100).astype('uint8')
            caffe_image.save_mean(mean_image, os.path.join(folder, 'mean.binaryproto'))

    return

if __name__ == '__main__':
    start_time = time.time()

    create_lmdbs()

    print 'Done after %s seconds' % (time.time() - start_time,)

