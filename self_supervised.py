'''
A set of functions to test out self supervised rotated coin image models
'''

import cPickle as pickle
import random
import glob
import os
import create_lmdb_rotate_whole_image
import summarize_whole_rotated_model_results
import caffe_image as ci
import pandas as pd
import shutil
import sys
import cv2

home_dir = '/home/pkrush/lmdb-files/'
data_dir = home_dir + 'metadata/'
crop_dir = home_dir + 'crops/'
train_dir = home_dir + 'train/'
test_dir = home_dir + 'test/'
index_filename = data_dir + 'index.p'


def create_index():
    index = [random.randint(4000, 13828) for x in range(250)]
    pickle.dump(index, open(index_filename, "wb"))


def get_index():
    index = pickle.load(open(index_filename, "rb"))
    return sorted(index)


def rename_crops():
    crops = []
    for filename in glob.iglob(crop_dir + '*.jpg'):
        crops.append([random.random(), filename])
    crops.sort()
    pickle.dump(crops, open(data_dir + 'copper_crops.p', "wb"))
    key = 0
    for rand, filename in crops:
        key += 1
        os.rename(filename, crop_dir + str(key) + '.jpg')


def copy_file(filename, dir):
    with open(filename, 'r') as myfile:
        data = myfile.read().replace('replace_dir_name_', dir)
    with open(dir + filename, 'w') as file_:
        file_.write(data)


def create_single_lmdbs(index):
    weight_filename = 'starting-weights.caffemodel'
    shutil.copyfile(weight_filename, train_dir + weight_filename)
    shell_filenames = []
    for image_id in index:
        filedata = [[image_id, crop_dir + str(image_id) + '.jpg', 0]]
        lmdb_dir = train_dir + str(image_id) + '/'
        create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 100,-1, True, False)
        print 'create single lmdb for ' + str(image_id)
        copy_train_files(lmdb_dir)
        shell_filename = create_train_script(image_id,lmdb_dir,train_dir + weight_filename)
        shell_filenames.append(shell_filename)
    create_script_calling_script(train_dir + 'train_all', shell_filenames)

def copy_train_files(lmdb_dir):
    copy_file('solver.prototxt', lmdb_dir)
    copy_file('train_val.prototxt', lmdb_dir)
    copy_file('deploy.prototxt', lmdb_dir)
    copy_file('labels.txt', lmdb_dir)


def create_train_script(image_id,lmdb_dir,weight_filename):
    shell_script = 'cd ' + lmdb_dir + '\n'
    shell_script += '/home/pkrush/caffe/build/tools/caffe '
    shell_script += 'train '

    shell_script += '-solver ' + lmdb_dir + 'solver.prototxt '
    shell_script += '-weights ' + weight_filename + ' '
    shell_script += '2> ' + lmdb_dir + 'caffe_output.log \n'
    shell_script += 'echo ' + str(image_id) + '\n'
    shell_filename = lmdb_dir + 'train-single-coin-lmdbs.sh'
    create_shell_script(shell_filename, shell_script)
    return shell_filename


def create_single_lmdb(seed_id):
    #weight_filename = train_dir + str(seed_id) + '/' + 'snapshot_iter_844.caffemodel'
    #weight_filename_copy = train_dir + 'snapshot_iter_844.caffemodel'
    #shutil.copyfile(weight_filename, weight_filename_copy)
    weight_filename_copy = 'snapshot_iter_844.caffemodel'

    seeds = pickle.load(open(data_dir + 'seed_data.pickle', "rb"))
    filedata = []
    values = seeds[seed_id]
    values.sort(key=lambda x: x[0], reverse=True)
    filedata.append([seed_id, crop_dir + str(seed_id) + '.jpg', 0])
    for max_value, angle, image_id in values:
        filedata.append([image_id, crop_dir + str(image_id) + '.jpg', angle])
    lmdb_dir = train_dir + str(seed_id) + '/'
    create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir,50, -1,True, False)
    copy_train_files(lmdb_dir)
    create_train_script(seed_id, lmdb_dir, weight_filename_copy)

def create_test_lmdbs(test_id):
    index = [x for x in range(4000)]
    filedata = []
    lmdb_dir = test_dir + str(test_id) + '/'
    for image_id in index:
        filedata.append([image_id, crop_dir + str(image_id) + '.jpg', 0])

    create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 3,test_id, False, False)

    shell_filenames = []
    index = get_index()

    for image_id in index:
        shell_script = 'cd ' + train_dir + str(image_id) + '/\n'
        shell_script += '/home/pkrush/caffe/.build_release/examples/cpp_classification/classification.bin '
        shell_script += 'deploy.prototxt '
        shell_script += 'snapshot_iter_844.caffemodel '
        shell_script += 'mean.binaryproto '
        shell_script += 'labels.txt '
        shell_script += test_dir + str(test_id) + '/train_db/data.mdb '
        shell_script += '> ' + test_dir + str(test_id) + '/' + str(image_id) + '.dat\n'
        shell_script += 'echo ' + str(image_id) + '\n'
        shell_filename = test_dir + str(test_id) + '/test-' + str(image_id) + '.sh'
        create_shell_script(shell_filename, shell_script)
        shell_filenames.append(shell_filename)
    create_script_calling_script(test_dir + 'test_all', shell_filenames)

def create_shell_script(filename, shell_script):
    shell_script = '#!/bin/bash\n' + shell_script
    with open(filename, 'w') as file_:
        file_.write(shell_script)

    fd = os.open(filename, os.O_RDONLY)
    os.fchmod(fd, 0755)
    os.close(fd)


def create_script_calling_script(filename, shell_filenames):
    shell_script = ''
    for shell_filename in shell_filenames:
        shell_script += shell_filename + '\n'
    create_shell_script(filename, shell_script)


def read_test(index,test_id,low_angle,high_angle):
    all_results  = pickle.load(open(data_dir +  'all_results.pickle', "rb"))
    new_all_results = []

    if len(index) == 1:
        for results in all_results:
            if results[0] != index[0]:
                new_all_results.append(results)

    for test in range(0,test_id + 1):
        print test
        for image_id in index:
            filename = test_dir + str(test) + '/' + str(image_id) + '.dat'

            if not os.path.isfile(filename):
                continue
            print filename
            results = summarize_whole_rotated_model_results.summarize_whole_rotated_model_results(filename, image_id,low_angle,high_angle)
            new_all_results.append(results)
    pickle.dump(new_all_results, open(data_dir + 'all_results.pickle', "wb"))


def read_all_results(cut_off):
    all_results = pickle.load(open(data_dir + 'all_results.pickle', "rb"))
    #columns = ['seed_image_id', 'key', 'angle', 'max_value']
    crops = {}
    seeds = {}
    for results in all_results:
        for seed_image_id, key, angle, max_value in results:
            if max_value < cut_off:
                continue
            if key in crops:
                if crops[key][2] < max_value:
                    crops[key] = [seed_image_id, angle, max_value]
            else:
                crops[key] = [seed_image_id, angle, max_value]

    for key, values in crops.iteritems():
        if not values[0] in seeds:
            seeds[values[0]] = []
        seeds[values[0]].append([values[2], values[1], key])

    for seed_image_id, values in seeds.iteritems():
        values.sort(key=lambda x: x[0], reverse=True)
        images = []
        square_size = 6
        count = 0
        crop_size = 120
        images.append(ci.get_rotated_crop(crop_dir,seed_image_id, crop_size, 0))
        for max_value, angle, image_id in values:
            crop = ci.get_rotated_crop(crop_dir,image_id, crop_size, angle)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(crop, str(max_value)[0:5], (10, 20), font, .7, (0, 255, 0), 2)
            images.append(crop)
        composite_image = ci.get_composite_image(images, square_size)
        cv2.imwrite(data_dir + str(seed_image_id) + '.png', composite_image)

    pickle.dump(seeds, open(data_dir + 'seed_data.pickle', "wb"))

#Instructions:
# create_index()
# create_single_lmdbs(get_index())
# in the train dir run ./train_all.sh
#create_test_lmdbs(1)
# in the test dir run ./test_all
#read_test(get_index())
#read_all_results()

#Single ReTrain & Test:
#create_single_lmdb(12004)
#in the train dir ./train-single-coin-lmdbs.sh
#in the test dir run ./test-12004.sh
#read_test([12004],1,60,300)
#in the metadata dir rm *.png
read_all_results(0)


