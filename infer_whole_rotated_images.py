'''
This infers caffe models using python.
This was abandon in favor of infering with the C++ program
'''


import os
import sys
import cv2
import random
import matplotlib.pyplot as plt
import infer
import caffe_image as ci

sys.path.append('/home/pkrush/caffe/python')
sys.path.append('/home/pkrush/digits')

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import numpy as np

if __name__ == '__main__':
    dirname = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(dirname, '..', '..'))
    import digits.config

# Import digits.config first to set the path to Caffe
import caffe.io
from caffe.proto import caffe_pb2

def infer_one_coin():
    crop_size = 28
    before_rotate_size = 100

    one_coin_rotated = infer.get_classifier("one_coin_rotated", crop_size)
    crop = cv2.imread('/home/pkrush/2-camera-scripts/crops/30287.png')
    if crop is None:
        return

    crop = cv2.resize(crop, (before_rotate_size, before_rotate_size), interpolation=cv2.INTER_AREA)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    mask = ci.get_circle_mask(crop_size)
    diff_angles = []
    scores = []

    for count1 in range(0, 100):
        random_angle = random.random() * 360
        class_angle = int(round(random_angle))
        rot_image = ci.get_whole_rotated_image(crop, mask, random_angle, crop_size)
        rot_image = ci.get_caffe_image(rot_image, crop_size)
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

