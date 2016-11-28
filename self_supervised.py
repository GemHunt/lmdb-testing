'''
A set of functions to test out self supervised rotated coin image models
'''

import cPickle as pickle
import random
import glob
import os
import create_lmdb_rotate_whole_image
import summarize_whole_rotated_model_results
import summarize_rotated_crops
import image_set
import shutil
import time
import subprocess


home_dir = '/home/pkrush/lmdb-files/'
data_dir = home_dir + 'metadata/'
crop_dir = home_dir + 'crops/'
train_dir = home_dir + 'train/'
test_dir = home_dir + 'test/'
test_angles = {0: (30, 330), 1: (60, 300), 2: (90, 270), 3: (120, 240), 4: (150, 210), 5: (180, 180)}


def init_dir():
    directories = [home_dir,data_dir,crop_dir,train_dir,test_dir]
    for test_id in range(0,6):
        directories.append(test_dir + str(test_id) + '/')
    make_dir(directories)

def make_dir(directories):
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)

def create_new_seed_index():
    seed_image_ids = [random.randint(4000, 13828) for x in range(250)]
    pickle.dump(seed_image_ids, open(data_dir + 'seed_image_ids.pickle', "wb"))

def get_seed_image_ids():
    return get_test_image_ids()

    #seed_image_ids = pickle.load(open(data_dir + 'seed_image_ids.pickle', "rb"))
    #return sorted(set(seed_image_ids))

    #test_image_ids = pickle.load(open(data_dir + 'test_image_ids.pickle', "rb"))
    #seed_image_ids = seed_image_ids + test_image_ids[0:180]
    #seed_image_ids = seed_image_ids + get_wide_image_ids()
    #pickle.dump(seed_image_ids, open(data_dir + 'seed_image_ids.pickle', "wb"))

def get_test_image_ids():
    test_image_ids = pickle.load(open(data_dir + 'test_image_ids.pickle', "rb"))
    return sorted(set(test_image_ids))

    #test_image_ids += get_seed_image_ids()
    #test_image_ids += get_wide_image_ids()
    #test_image_ids = list(set(test_image_ids))
    #pickle.dump(test_image_ids, open(data_dir + 'test_image_ids.pickle', "wb"))

def get_wide_image_ids():
    return set([11458,12004])

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


def create_single_lmdbs(seed_image_ids):
    weight_filename = 'starting-weights.caffemodel'
    shutil.copyfile(weight_filename, train_dir + weight_filename)
    shell_filenames = []
    for image_id in seed_image_ids:
        filedata = [[image_id, crop_dir + str(image_id) + '.jpg', 0]]
        lmdb_dir = train_dir + str(image_id) + '/'
        create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 100,-1, True, False)
        print 'Creating single lmdb for ' + str(image_id)
        copy_train_files(lmdb_dir)
        shell_filename = create_train_script(image_id,lmdb_dir,train_dir + weight_filename)
        shell_filenames.append(shell_filename)
    create_script_calling_script(train_dir + 'train_all.sh', shell_filenames)

def copy_train_files(lmdb_dir):
    copy_file('solver.prototxt', lmdb_dir)
    copy_file('train_val.prototxt', lmdb_dir)
    copy_file('deploy.prototxt', lmdb_dir)
    copy_file('labels.txt', lmdb_dir)


def create_train_script(image_id,lmdb_dir,weight_filename):
    log_filename = 'caffe_output.log'
    shell_script = 'cd ' + lmdb_dir + '\n'
    shell_script += '/home/pkrush/caffe/build/tools/caffe '
    shell_script += 'train '
    shell_script += '-solver ' + lmdb_dir + 'solver.prototxt '
    shell_script += '-weights ' + weight_filename + ' '
    shell_script += '2> ' + lmdb_dir + log_filename + ' \n'
    shell_script += 'grep accu ' + log_filename + ' \n'
    shell_filename = lmdb_dir + 'train-single-coin-lmdbs.sh'
    create_shell_script(shell_filename, shell_script)
    return shell_filename


def create_single_lmdb(seed_id):
    start_time = time.time()
    print 'create_single_lmdb for ' + str(seed_id)

    weight_filename = train_dir + str(seed_id) + '/' + 'snapshot_iter_844.caffemodel'
    weight_filename_copy = train_dir + 'snapshot_iter_844.caffemodel'
    shutil.copyfile(weight_filename, weight_filename_copy)
    #weight_filename_copy = 'snapshot_iter_844.caffemodel'

    seeds = pickle.load(open(data_dir + 'seed_data.pickle', "rb"))
    filedata = []
    values = seeds[seed_id]
    values.sort(key=lambda x: x[0], reverse=True)

    #this is handy for large groups (heads,tails)
    #best_results_by_angle_group = {}
    #for max_value, angle, image_id in values:
        #rounded_angle = int(round(angle / 5) * 5)
        #if not rounded_angle in best_results_by_angle_group.keys():
            #best_results_by_angle_group[rounded_angle] = [max_value, angle, image_id]
        #else:
            #if max_value > best_results_by_angle_group[rounded_angle][0]:
                #best_results_by_angle_group[rounded_angle] = [max_value, angle, image_id]
    #values = best_results_by_angle_group.values()

    filedata.append([seed_id, crop_dir + str(seed_id) + '.jpg', 0])
    for max_value, angle, image_id in values:
        if max_value > 20:
            filedata.append([image_id, crop_dir + str(image_id) + '.jpg', angle])
    lmdb_dir = train_dir + str(seed_id) + '/'

    create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir,50, -1,True, False)
    copy_train_files(lmdb_dir)
    create_train_script(seed_id, lmdb_dir, weight_filename_copy)
    print 'Done in %s seconds' % (time.time() - start_time,)


def create_test_lmdbs(test_id):
    #test_image_ids = [x for x in range(13927)]
    test_image_ids = get_test_image_ids()
    filedata = []
    lmdb_dir = test_dir + str(test_id) + '/'
    for image_id in test_image_ids:
        filedata.append([image_id, crop_dir + str(image_id) + '.jpg', 0])

    create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 3,test_id, False, False)

    shell_filenames = []
    seed_image_ids = get_seed_image_ids()

    for image_id in seed_image_ids:
        shell_script = 'cd ' + train_dir + str(image_id) + '/\n'
        shell_script += '/home/pkrush/caffe/.build_release/examples/cpp_classification/classification.bin '
        shell_script += 'deploy.prototxt '
        shell_script += 'snapshot_iter_844.caffemodel '
        shell_script += 'mean.binaryproto '
        shell_script += 'labels.txt '
        shell_script += test_dir + str(test_id) + '/train_db/data.mdb '
        shell_script += '> ' + test_dir + str(test_id) + '/' + str(image_id) + '.dat\n'
        shell_filename = test_dir + str(test_id) + '/test-' + str(image_id) + '.sh'
        create_shell_script(shell_filename, shell_script)
        shell_filenames.append(shell_filename)
    create_script_calling_script(test_dir + 'test_all.sh', shell_filenames)

def create_shell_script(filename, shell_script):
    shell_script = '#!/bin/bash\n' + 'echo Entered ' + filename + '\n' + shell_script
    shell_script = shell_script + 'echo Exited ' + filename + '\n'

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

def read_test(image_ids, max_test_id):
    all_results_filename = data_dir +  'all_results.pickle'
    all_results =[]
    new_all_results = []
    if os.path.exists(all_results_filename):
        all_results  = pickle.load(open(data_dir +  'all_results.pickle', "rb"))

    #If only one image is being read remove the old image, else output all new results
    if len(image_ids) == 1:
        for results in all_results:
            if results[0] != image_ids[0]:
                new_all_results.append(results)

    low_angle, high_angle = test_angles[max_test_id]
    for test_id in range(0,max_test_id + 1):
        for image_id in image_ids:
            filename = test_dir + str(test_id) + '/' + str(image_id) + '.dat'

            if not os.path.isfile(filename):
                continue
            results = summarize_rotated_crops.get_results(filename, image_id,low_angle,high_angle)
            #results = summarize_whole_rotated_model_results.summarize_whole_rotated_model_results(filename, image_id,low_angle,high_angle)
            new_all_results.append(results)
    pickle.dump(new_all_results, open(data_dir + 'all_results.pickle', "wb"))

def widen_model(seed_image_id):
    for test_id in range(1,6):
        create_single_lmdb(seed_image_id)
        run_script(train_dir + str(seed_image_id) + '/train-single-coin-lmdbs.sh')
        run_script(test_dir + str(test_id) + '/test-'+ str(seed_image_id) + '.sh')
        read_test([seed_image_id],test_id)
        #in the metadata dir rm *.png
        read_all_results(20)

def create_all_test_lmdbs():
    for test_id in range(1, 6):
        create_test_lmdbs(test_id)

def test_all(seed_image_ids):
    for seed_image_id in seed_image_ids:
        for test_id in range(0, 6):
            run_script(test_dir + str(test_id) + '/test-' + str(seed_image_id) + '.sh')

    read_test(seed_image_ids, 5)
    read_all_results(0,seed_image_ids)

def run_script(filename):
    print "Running " + filename
    subprocess.call(filename)


def create_new_indexes(total_new_seed_imgs,total_new_test_imgs):
    seeds = pickle.load(open(data_dir + 'seed_data.pickle', "rb"))
    seed_image_ids = []
    test_image_ids = []
    count = 0

    for seed_image_id, values in seeds.iteritems():
        values.sort(key=lambda x: x[0], reverse=False)
        #seed_image_ids.append(values[0:total_new_seed_imgs][2])
        #test_image_ids.append(values[total_new_seed_imgs:total_new_seed_imgs+total_new_test_imgs][2])

        for max_value, angle, image_id in values:
            count += 1
            if count < total_new_seed_imgs:
                seed_image_ids.append(image_id)
            else:
                if count < total_new_seed_imgs + total_new_test_imgs:
                    test_image_ids.append(image_id)
        count = 0

    pickle.dump(seed_image_ids, open(data_dir + 'seed_image_ids.pickle', "wb"))
    pickle.dump(test_image_ids, open(data_dir + 'test_image_ids.pickle', "wb"))


#Instructions from scratch:
#create_new_seed_index()
#seeds = get_seed_image_ids()-get_wide_image_ids()
#create_single_lmdbs(seeds)
#create_test_lmdbs(0)
#run_script(train_dir + 'train_all.sh')
#run_script(test_dir + 'test_all.sh')
#read_test(get_seed_image_ids(),0)
#read_all_results(10)


#Pick top seed with the most image results over 20 and highest of those results:
#widen_model(3360)

#Shrink the results to the widened seeds:
#read_all_results(0,[11458,12004])

#create_all_test_lmdbs()  #Raise the number of test images

#test_all(seed_image_ids)

#Check out the test set results and choose the number of seeds(60) and training images(1000).
#Test on new test set and make 30 new seeds low performers of each set.
#Create test sets from the 500 lowest performers of each set.
#create_new_indexes(30, 500)


def read_all_results(cut_off = 0,seed_image_ids = [], many_image_ids_per_seed_ok = True):
    image_set.read_results(cut_off,data_dir,seed_image_ids)
    #image_set.create_composite_images(crop_dir, data_dir, 120,5,5)
    image_set.set_angles_postive()
    image_set.set_starting_seed()



#Second Try Script:
#I renamed lmdb-files to lmdbfiles100
#I also copied the crops and 2 pickles for seeds and test IDs
#I cropped 56x56 from center, dropped using the mask, and dropped the resize.
#init_dir()
#seeds = get_seed_image_ids()
#create_single_lmdbs(seeds)
#create_test_lmdbs(0)
#run_script(train_dir + 'train_all.sh')
#run_script(test_dir + 'test_all.sh')
#read_test(get_seed_image_ids(),0)
read_all_results(15)



