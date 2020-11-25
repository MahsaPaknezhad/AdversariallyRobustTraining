#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torchvision
import torchvision.transforms as transforms
from DatasetTemplate import DatasetTemplate

class DatasetMNIST(DatasetTemplate):
    def __init__(self, param):
        DatasetTemplate.__init__(self)

        self.param = param
        self.mean = [0.1307]
        self.std = [0.3081]

        self.transform_train = transforms.Compose([
                                    transforms.Resize((param.imsize, param.imsize)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(self.mean, self.std)
                                    ])

        self.transform_val = transforms.Compose([
                                transforms.Resize((param.imsize, param.imsize)),
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

        self.supported_models = ['BasicModel']

    def getTrainVal(self, train_val_ratio=0.8):
        param = self.param
        np.random.seed(param.seed)

        trainset = torchvision.datasets.MNIST(root=param.data_dir, train=True, download=True, transform=self.transform_train)
        
        ###Splitting into training set and validation set
        x = np.asarray(trainset.data)
        y = np.asarray(trainset.targets)
        
        ###Select index of each class
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
        
        return (x_train, y_train), (x_val, y_val)
    
    def getTest(self):
        param = self.param
        np.random.seed(param.seed)

        testset = torchvision.datasets.MNIST(root=param.data_dir, train=False, download=True, transform=self.transform_test)
        x_test = np.asarray(testset.data)
        y_test = np.asarray(testset.targets)

        return (x_test, y_test)
