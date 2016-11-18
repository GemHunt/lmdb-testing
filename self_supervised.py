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
    index = get_index()
    weight_filename = 'starting-weights.caffemodel'
    shutil.copyfile(weight_filename, train_dir + weight_filename)
    shell_filenames = []
    for image_id in index:
        filedata = [[image_id, crop_dir + str(image_id) + '.jpg', 0]]
        lmdb_dir = train_dir + str(image_id) + '/'
        #create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 100,True,False)
        copy_file('solver.prototxt', lmdb_dir)
        copy_file('train_val.prototxt', lmdb_dir)
        copy_file('deploy.prototxt', lmdb_dir)
        copy_file('labels.txt', lmdb_dir)
        print 'create single lmdb for ' + str(image_id)
        shell_script = 'cd ' + lmdb_dir + '\n'
        shell_script += '/home/pkrush/caffe/build/tools/caffe '
        shell_script += 'train '

        shell_script += '-solver ' + lmdb_dir + 'solver.prototxt '
        shell_script += '-weights ' + train_dir + weight_filename + ' '
        shell_script += '2> ' + lmdb_dir + 'caffe_output.log \n'

        shell_filename = lmdb_dir + 'train-single-coin-lmdbs.sh'
        shell_filenames.append(shell_filename)
        create_shell_script(shell_filename, shell_script)

    create_script_calling_script(train_dir + 'train_all', shell_filenames)

def create_test_lmdbs():
    index = [x for x in range(360)]
    filedata = []
    lmdb_dir = test_dir + str(0) + '/'
    for image_id in index:
        filedata.append([image_id, crop_dir + str(image_id) + '.jpg', 0])

    #create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 3,False,False)

    shell_filenames = []
    index = get_index()

    for image_id in index:
        shell_script = 'cd ' + train_dir + str(image_id) + '/\n'
        shell_script += '/home/pkrush/caffe/.build_release/examples/cpp_classification/classification.bin '
        shell_script += 'deploy.prototxt '
        shell_script += 'snapshot_iter_844.caffemodel '
        shell_script += 'mean.binaryproto '
        shell_script += 'labels.txt '
        shell_script += test_dir + '0/train_db/data.mdb '
        shell_script += '> ' + test_dir + '0/' + str(image_id) + '.dat\n'
        shell_script += 'echo ' + str(image_id) + '\n'
        shell_filename = test_dir + '0/test-' + str(image_id) + '.sh'
        shell_filenames.append(shell_filename)
        create_shell_script(shell_filename, shell_script)

    create_script_calling_script(test_dir + 'test_all', shell_filenames)


def create_shell_script(filename,shell_script):
    shell_script = '#!/bin/bash\n' + shell_script
    with open(filename, 'w') as file_:
        file_.write(shell_script)

    fd = os.open(filename, os.O_RDONLY)
    os.fchmod(fd, 0755)
    os.close(fd)

def create_script_calling_script(filename,shell_filenames):
    shell_script = ''
    for shell_filename in shell_filenames:
        shell_script += shell_filename + '\n'
    create_shell_script(filename, shell_script)


def read_test():
    index = get_index()
    all_results = []
    for image_id in index:
        filename = test_dir + '0/' + str(image_id) + '.dat'
        results = summarize_whole_rotated_model_results.summarize_whole_rotated_model_results(filename, image_id)
        all_results.append(results )
        images = []
        square_size = 5
        for count in range(0,square_size * square_size):
            if count +2 > len(results):
                break
            if count == 0:
                angle = 0
                crop_id = image_id
            else:
                crop_id = results[count - 1][1]
                angle = results[count - 1][2]

            crop = cv2.imread(crop_dir + str(crop_id) + '.jpg')
            crop = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_AREA)
            M = cv2.getRotationMatrix2D((50, 50), angle, 1)
            cv2.warpAffine(crop, M, (100, 100), crop, cv2.INTER_CUBIC)
            images.append(crop)
        composite_image = ci.get_composite_image(images,square_size)
        cv2.imwrite(test_dir  + '0/' + str(image_id) + '.png',composite_image)
    pickle.dump(all_results, open(test_dir + '0/all_results.pickle', "wb"))


def read_all_results():
    all_results = pickle.load( open(test_dir + '0/all_results.pickle' , "rb" ) )
    #idx = all_results.groupby(['index_id'])['max_value'].transform(max) == all_results['max_value']
    #print all_results[idx]
    print all_results


###Instructions:
#create_index()
#create_single_lmdbs()
#in the train dir run ./train_all.sh
#create_test_lmdbs()
#in the test dir run ./test_all
#read_test()
read_all_results()

