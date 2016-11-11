'''
A set of functions to test out self supervised rotated coin image models
'''

import cPickle as pickle
import random
import glob
import os
import create_lmdb_rotate_whole_image
import shutil
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
    index = pickle.load( open(index_filename , "rb" ) )
    return sorted(index)

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

def copy_file(filename,dir):
    with open(filename, 'r') as myfile:
        data = myfile.read().replace('replace_dir_name_', dir)
    with open(dir + filename, 'w') as file_:
        file_.write(data)

def create_single_lmdbs():
    shell_script = '#!/bin/bash\n'
    index = get_index()
    weight_filename = 'starting-weights.caffemodel'
    shutil.copyfile(weight_filename, train_dir + weight_filename)
    shutil.copyfile('deploy.prototxt', train_dir + 'deploy.prototxt')
    for image_id in index:
        filedata = [[image_id, crop_dir + str(image_id) + '.jpg', 0]]
        lmdb_dir = train_dir + str(image_id) + '/'
        create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 100)
        copy_file('solver.prototxt', lmdb_dir)
        copy_file('train_val.prototxt', lmdb_dir)
        print 'create single lmdb for ' + str(image_id)
        shell_script += '/home/pkrush/caffe/build/tools/caffe '
        shell_script += 'train '
        shell_script += '-solver ' + lmdb_dir + 'solver.prototxt '
        shell_script += '-weights ' + train_dir + weight_filename + ' '
        shell_script += '2> ' + lmdb_dir + 'caffe_output.log \n'
    with open(train_dir + 'train-single-coin-lmdbs.sh', 'w') as file_:
        file_.write(shell_script)

def create_test_lmdbs():
    index = [x for x in range(10)]
    filedata = []
    for image_id in index:
        filedata.append([image_id, crop_dir + str(image_id) + '.jpg', 0])

    lmdb_dir = test_dir + str(0) + '/'
    create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 10,create_val_set = False)
    print 'create single lmdb for ' + str(image_id)




create_test_lmdbs()
