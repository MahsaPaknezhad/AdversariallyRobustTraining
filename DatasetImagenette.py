#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility class to prepare the train, validation and test splits of the Imagenette dataset.

import os
import torch
import numpy as np
import torchvision.transforms as transforms

from AutoAugment import ImageNetPolicy
from DatasetTemplate import DatasetTemplate
from PIL import Image
from Logging import info, success

class DatasetImagenette(DatasetTemplate):
    def __init__(self, param):
        super(DatasetImagenette, self).__init__()
        self.param = param
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.transform_train = transforms.Compose([
                                    transforms.RandomResizedCrop(self.param.imsize, (0.8, 1.0)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomPerspective(p = 0.75, distortion_scale=0.2),
                                    ImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(self.mean, self.std),
                                    ])
        
        self.transform_val = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(self.mean, self.std),
                                    ])
        if "softmax_plotting_mode" in param:
            self.transform_test = transforms.Compose([
                                        transforms.ToTensor()
                                        ])
        else:
            self.transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)
                                        ])

        self.supported_models = ['ResNet18', 'XResNet18', 'XResNet34', 'XResNet50', 'XResNet18Mod']

        self.nameToClass = dict([('n01440764', 'tench'),
                                 ('n02102040', 'english springer'),
                                 ('n02979186', 'casette player'),
                                 ('n03000684', 'chainsaw'),
                                 ('n03028079', 'church'),
                                 ('n03394916', 'french horn'),
                                 ('n03417042', 'garbage truck'),
                                 ('n03425413', 'gas pump'),
                                 ('n03445777', 'golf ball'),
                                 ('n03888257', 'parachute')])

        self.nameToID = dict([('n01440764', 0),
                              ('n02102040', 1),
                              ('n02979186', 2),
                              ('n03000684', 3),
                              ('n03028079', 4),
                              ('n03394916', 5),
                              ('n03417042', 6),
                              ('n03425413', 7),
                              ('n03445777', 8),
                              ('n03888257', 9)])

        self.classToName = {}
        for i in range(len(self.nameToClass)):
            self.classToName[list(self.nameToClass.values())[i]] = list(self.nameToClass.keys())[i]
    
    def getTrainVal(self, fast_mode = True, train_val_ratio=0.8):
        param = self.param
        size = param.imsize

        np.random.seed(param.seed)
        
        x_train = []
        y_train = []
        
        x_val = []
        y_val = []
        
        main_path = os.path.join(param.data_dir, 'imagenette')
        train_path = os.path.join(main_path, 'train')

        info('Commencing dataset preparation.')

        if fast_mode:
            try:
                x = np.load(os.path.join(main_path, 'x_train{}.npy'.format(size)))
                y = np.load(os.path.join(main_path, 'y_train{}.npy'.format(size)))
            except:
                raise Exception("No numpy files found in given data directory. Run PrepImagenette.py from ImagenetteUtils first")
            
            train_idx = np.array([])
            val_idx = np.array([])
            for i in range(param.num_class):
                idx = np.where(y == i)[0]
                np.random.shuffle(idx)
                
                t_idx = idx[:int(idx.shape[0]*train_val_ratio)]
                v_idx = idx[int(idx.shape[0]*train_val_ratio):]

                if param.num_train != -1 and param.num_val != -1:

                    train_idx = np.append(train_idx, t_idx[:param.num_train]).astype(np.int)
                    val_idx = np.append(val_idx, v_idx[:param.num_val]).astype(np.int)
                else:

                    train_idx = np.append(train_idx, t_idx).astype(np.int)
                    val_idx = np.append(val_idx, v_idx).astype(np.int)

            x_train = x[train_idx]
            y_train = y[train_idx]
            
            x_val = x[val_idx]
            y_val = y[val_idx]

        else:
            for label in os.listdir(train_path):
                img_list = os.listdir(os.path.join(train_path, label))
                idxes = np.arange(0, len(img_list))
                train_id = np.random.choice(idxes, param.num_train, False)
                val_id = np.array(list(set(idxes) - set(train_id)))
                val_id = np.random.choice(val_id, param.num_val, False)
                
                for idx in train_id:
                    im = Image.open(os.path.join(train_path, label, img_list[idx]))
                    im = im.resize((size, size)) #Shape: WHC, which is correct
                    im = np.array(im)
                    if im.shape != (size, size, 3):
                        im = np.expand_dims(im, axis = -1).repeat(3, axis = -1)
                    x_train.append(im)
                    y_train.append(self.nameToID[label])
                
                for idx in val_id:
                    im = Image.open(os.path.join(train_path, label, img_list[idx]))
                    im = im.resize((size, size)) #Shape: WHC, which is correct
                    im = np.array(im)
                    if im.shape != (size, size, 3):
                        im = np.expand_dims(im, axis = -1).repeat(3, axis = -1)
                    x_val.append(im)
                    y_val.append(self.nameToID[label])
                    
        success('Dataset preparation complete.')

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        assert x_train.shape[0] == y_train.shape[0]
        
        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)
        assert x_val.shape[0] == y_val.shape[0]
        
        return (x_train, y_train), (x_val, y_val)

    def getTest(self, fast_mode = True):
        param = self.param
        size = param.imsize

        np.random.seed(param.seed)
        
        x_test = []
        y_test = []
        
        main_path = os.path.join(param.data_dir, 'imagenette')
        test_path = os.path.join(main_path, 'val') # 'Test' set is named 'val' 
        
        info('Preparing test split of dataset.')

        if fast_mode:
            try:
                x_test = np.load(os.path.join(main_path, 'x_val128.npy'))
                if "softmax_plotting_mode" in param:
                    if param.softmax_plotting_mode:
                        x_test_new = torch.zeros((x_test.shape[0], size, size, 3), dtype=torch.uint8)
                        for index, img in enumerate(x_test):
                            resized_img = np.resize(img, (size, size, 3))
                            x_test_new[index] = torch.tensor(resized_img, dtype=torch.uint8)
                        x_test = x_test_new
                y_test = np.load(os.path.join(main_path, 'y_val128.npy'))
            except:
                raise Exception("No numpy files found in given data_dir. Run prepImagenette.py from imagenetteUtils first")
        
        else:
            for label in os.listdir(test_path):
                img_list = os.listdir(os.path.join(test_path, label))
                test_id = np.arange(0, len(img_list))
        
                for idx in test_id:
                    im = Image.open(os.path.join(test_path, label, img_list[idx]))
                    im = im.resize((size, size)) #Shape: WHC, which is correct
                    im = np.array(im)
                    if im.shape != (size, size, 3):
                        im = np.expand_dims(im, axis = -1).repeat(3, axis = -1)
                    x_test.append(im)
                    y_test.append(self.nameToID[label])
            
        success('Dataset test split preparation complete.')

        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        assert x_test.shape[0] == y_test.shape[0]

        return (x_test, y_test)
            
    def folder_class(self, foldername):
        return self.nameToClass[foldername]

    def class_folder(self, classname):
        return self.classToName[classname]