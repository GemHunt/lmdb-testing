'''
A set of functions to test out self supervised rotated coin image models
'''

import cPickle as pickle
import random
import glob
import os

data_dir = '/home/pkrush/lmdb-files/data/'
crop_dir = '/home/pkrush/lmdb-files/crops/'
index_filename = data_dir + 'index.p'


def create_index():
    index = [random.randint(1000, 14321) for x in range(25)]
    pickle.dump(index, open(index_filename , "wb") )

def get_index():
    index = pickle.load( open(index_filename , "rb" ) )

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


rename_crops():
    pass



