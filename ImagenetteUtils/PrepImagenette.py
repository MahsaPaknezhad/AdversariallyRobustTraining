#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility file to ensure that the Imagenette dataset follows certain guidelines to make dataset preparation faster.
# This code assumes that the Imagenette dataset has been downloaded from https://github.com/fastai/imagenette and placed the data directory.

import os
import numpy as np

from PIL import Image
from Logging import info, success

size = 256
data_dir = '../data'

main_path = os.path.join(data_dir, 'imagenette')
train_path = os.path.join(main_path, 'train')
val_path = os.path.join(main_path, 'val')

info('Preparing dataset')

x_train = []
y_train = []
x_val = []
y_val = []

nameToID = dict([('n01440764', 0),
                 ('n02102040', 1),
                 ('n02979186', 2),
                 ('n03000684', 3),
                 ('n03028079', 4),
                 ('n03394916', 5),
                 ('n03417042', 6),
                 ('n03425413', 7),
                 ('n03445777', 8),
                 ('n03888257', 9)])
                
for label in os.listdir(train_path): # Same for both train and test
    '''
    For Training Images
    '''
    img_list = os.listdir(os.path.join(train_path, label))
    idxes = np.arange(0, len(img_list))
    
    for idx in idxes:
        im = Image.open(os.path.join(train_path, label, img_list[idx]))
        im = im.resize((size, size)) #Shape: WHC, which is correct
        im = np.array(im)
        if im.shape != (size, size, 3):
            im = np.expand_dims(im, axis = -1).repeat(3, axis = -1)
        x_train.append(im)
        y_train.append(nameToID[label])
    
    '''
    For Test Images
    '''
    img_list = os.listdir(os.path.join(val_path, label))
    idxes = np.arange(0, len(img_list))
    
    for idx in idxes:
        im = Image.open(os.path.join(val_path, label, img_list[idx]))
        im = im.resize((size, size)) #Shape: WHC, which is correct
        im = np.array(im)
        if im.shape != (size, size, 3):
            im = np.expand_dims(im, axis = -1).repeat(3, axis = -1)
        x_val.append(im)
        y_val.append(nameToID[label])
        
success('Dataset preparation completed.')

x_train = np.asarray(x_train)
y_train = np.array(y_train)
assert x_train.shape[0] == y_train.shape[0]

x_val = np.asarray(x_val)
y_val = np.array(y_val)
assert x_val.shape[0] == y_val.shape[0]


np.save(os.path.join(main_path, 'x_train{}.npy'.format(size)), x_train)
np.save(os.path.join(main_path, 'y_train{}.npy'.format(size)), y_train)
np.save(os.path.join(main_path, 'x_val{}.npy'.format(size)), x_val)
np.save(os.path.join(main_path, 'y_val{}.npy'.format(size)), y_val)


