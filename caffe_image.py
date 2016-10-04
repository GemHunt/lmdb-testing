import os
import sys
import cv2
import math
sys.path.append('/home/pkrush/caffe/python')
sys.path.append('/home/pkrush/digits')

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import numpy as np
import PIL.Image

if __name__ == '__main__':
    dirname = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(dirname,'..','..'))
    import digits.config

from digits import utils

# Import digits.config first to set the path to Caffe
import caffe.io
from caffe.proto import caffe_pb2


def rotate(img, angle):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    cv2.warpAffine(img, M, (cols, rows),img, cv2.INTER_CUBIC)
    return img

def rotate(img, angle,center_x,center_y,rows,cols):
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    cv2.warpAffine(img, M, (cols, rows),img, cv2.INTER_CUBIC)
    return img

def rotate_point(angle, center_x,center_y,point_x,point_y):
    rotated_x = ((point_x - center_x) * math.cos(angle)) - ((point_y - center_y) * math.sin(angle)) + center_x;
    rotated_y = ((point_x - center_x) * math.sin(angle)) + ((point_y - center_y) * math.cos(angle)) + center_y;
    return rotated_x,rotated_y


def rotate_matrix(angle, center_x,center_y,mat):
    rotated = mat.copy()
    #OMG I need to learny my matrix math in Python!
    for num in range(0,4):
        rotated_x,rotated_y = rotate_point(math.radians(angle),center_x,center_y,mat[num,0],mat[num,1])
        rotated[num,0] = rotated_x
        rotated[num, 1] = rotated_y
    return rotated

def get_angled_crops(crop, crop_size):
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (crop_size , crop_size), interpolation=cv2.INTER_AREA)

    crops = [None] * 360

    for angle in range(0, 360):
        # rotated = rotate(crop.copy(),angle)
        pts1 = np.float32([[440, 295], [595, 290], [440, 305], [595, 310]])
        rotated_mat = rotate_matrix(angle, 300, 300, pts1)

        pts2 = np.float32([[0, 0], [28, 0], [0, 14], [28, 14]])
        M = cv2.getPerspectiveTransform(rotated_mat, pts2)
        dst = cv2.warpPerspective(crop, M, (28, 14))

        crops[angle] = dst

    return crops

def save_image(image, filename):
    # converting from BGR to RGB
    image = image[[2,1,0],...] # channel swap
    #convert to (height, width, channels)
    image = image.astype('uint8').transpose((1,2,0))
    image = PIL.Image.fromarray(image)
    image.save(filename)

def save_mean(mean, filename):
    """
    Saves mean to file

    Arguments:
    mean -- the mean as an np.ndarray
    filename -- the location to save the image
    """
    if filename.endswith('.binaryproto'):
        blob = caffe_pb2.BlobProto()
        blob.num = 1
        blob.channels = mean.shape[0]
        blob.height = mean.shape[1]
        blob.width = mean.shape[2]
        blob.data.extend(mean.astype(float).flat)
        with open(filename, 'wb') as outfile:
            outfile.write(blob.SerializeToString())

    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        save_image(mean, filename)
    else:
        raise ValueError('unrecognized file extension')

    def get_caffe_image(crop, crop_size):
        # this is how you get the image from file:
        # coinImage = [caffe.io.load_image("some file", color=False)]

        caffe_image = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        caffe_image = caffe_image.astype(np.float32) / 255
        caffe_image = np.array(caffe_image).reshape(crop_size, crop_size, 1)
        # Caffe wants a list so []:
        caffe_images = [caffe_image]
        return caffe_images