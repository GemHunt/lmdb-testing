'''
A set of functions to test out self supervised rotated coin image models
'''

import cPickle as pickle
import random
import glob
import os
import create_lmdb_rotate_whole_image
import sys

home_dir = '/home/pkrush/lmdb-files/'
data_dir = home_dir + 'metadata/'
crop_dir = home_dir + 'crops/'
train_dir = home_dir + 'train/'
test_dir = home_dir + 'test/'
index_filename = data_dir + 'index.p'


def create_index():
    index = [random.randint(1000, 13828) for x in range(25)]
    pickle.dump(index, open(index_filename , "wb") )

def get_index():
    return pickle.load( open(index_filename , "rb" ) )


def rename_crops():
    crops = []
    for filename in glob.iglob(crop_dir + '*.jpg'):
        crops.append([random.random(),filename])
    crops.sort()
    pickle.dump(crops, open(data_dir + 'copper_crops.p', "wb"))
    key = 0
    for rand,filename in crops:
        key += 1
        os.rename(filename, crop_dir + str(key) + '.jpg')

def create_single_lmdbs():
    index = get_index()
    for image_id in index:
        filedata = [[image_id, crop_dir + str(image_id) + '.jpg', 0]]
        lmdb_dir = train_dir + str(image_id) + '/'
        create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 100)
        print 'create single lmdb for ' + str(image_id)

def create_test_lmdbs():
    index = [x for x in range(1000)]
    filedata = []
    for image_id in index:
        filedata.append([image_id, crop_dir + str(image_id) + '.jpg', 0])

    lmdb_dir = test_dir + str(0) + '/'
    create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 10,create_val_set = False)
    print 'create single lmdb for ' + str(image_id)



create_single_lmdbs()
