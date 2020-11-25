#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:08:05 2020

@author: ngopc
"""
import numpy as np
import torchvision
import torchvision.transforms as transforms

from DatasetTemplate import DatasetTemplate
from AutoAugment import CIFAR10Policy

class DatasetCIFAR10(DatasetTemplate):
    def __init__(self, param):
        DatasetTemplate.__init__(self)
        
        self.param = param
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.247, 0.243, 0.261]
        
        self.transform_train = transforms.Compose([
                                    transforms.RandomResizedCrop(param.imsize, (0.8, 1.0)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomPerspective(p = 0.75, distortion_scale=0.2),
                                    CIFAR10Policy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(self.mean, self.std),
                                    ])
        
        self.transform_val = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(self.mean, self.std),
                                    ])
        if "softmax_plotting_mode" in param:
            self.transform_test = transforms.Compose([
                                        transforms.Resize((param.imsize, param.imsize)),
                                        transforms.ToTensor()
                                        ])
        else:
            self.transform_test = transforms.Compose([
                                        transforms.Resize((param.imsize, param.imsize)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)
                                        ])
        
        self.supported_models = ["ResNet18", "ResNet9", "ResNet9_mod"]
    
    def getTrainVal(self, train_val_ratio=0.8):
        param = self.param
        np.random.seed(param.seed)
        
        trainset = torchvision.datasets.CIFAR10(root=param.data_dir, train=True, download=True, transform=None)
        
        ###Splitting into training set and validation set
        x = np.asarray(trainset.data)
        y = np.asarray(trainset.targets)
        
        ###Select index of each class

        train_idx = np.array([])
        val_idx = np.array([])
        for i in range(param.num_class):
            idx = np.where(y == i)[0]
            np.random.shuffle(idx)

            t_idx = idx[:int(idx.shape[0] * train_val_ratio)]
            v_idx = idx[int(idx.shape[0] * train_val_ratio):]

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

        return (x_train, y_train), (x_val, y_val)
    
    def getTest(self):
        param = self.param
        np.random.seed(param.seed)

        testset = torchvision.datasets.CIFAR10(root=param.data_dir, train=False, download=True, transform=None)
        x_test = np.asarray(testset.data)
        y_test = np.asarray(testset.targets)

        return (x_test, y_test)
