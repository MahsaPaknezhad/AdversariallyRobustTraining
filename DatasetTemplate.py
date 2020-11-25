#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:03:05 2020

@author: ngopc
"""
import numpy as np
import torchvision.transforms as transforms
from LoadModel import loadModel 

class DatasetTemplate:
    #You need to override this __init__ method
    def __init__(self):
        self.param = {} #param dictionary
        self.mean = [0] #dataset mean for normalization
        self.std = [1]  #dataset std for normalization
        
        self.transform_train = transforms.Compose([   #torch transformation for train set
                                    transforms.ToTensor(),
                                    transforms.Normalize(self.mean, self.std),
                                    ])
        self.transform_val = transforms.Compose([])   #torch transformation for val set
        self.transform_test = transforms.Compose([])  #torch transformation for test set 
        self.supported_models = [] #model name that is in folder models

    #You need to override this method
    def getTrainVal(self):
        print("You have not override getTrainVal() method in DatasetTemplate")
        print("Check getTrainVal() in DataTemplate to know how to write your own getTrainVal() method")
        
        #####Example Code######
        #Shape of x_train, x_val is [N, W, H, C]
        x_train = np.ones(shape=(100, 32, 32, 3))  #x_train contains integer in range [0,255]
        y_train = np.ones(shape=(100,))            #y_train contains integer in range [0, number of classes - 1]
        
        x_val = np.ones(shape=(100, 32, 32, 3))  #x_val contains integer in range [0,255]
        y_val = np.ones(shape=(100,))            #y_val contains integer in range [0, number of classes - 1]
        
        raise Exception("You have not override getTrainVal() method in Dataset class that inherits from DatasetTemplate")
        return (x_train, y_train), (x_val, y_val)
    
    #You need to override this method
    def getTest(self):
        print("You have not override getTest() method in DatasetTemplate")
        print("Check getTest() in DataTemplate to know how to write your own getTest() method")
        
        #####Example Code######
        #Shape of x_test is [N, W, H, C]
        x_test = np.ones(shape=(100, 32, 32, 3))  #x_test contains integer in range [0,255]
        y_test = np.ones(shape=(100,))            #y_test contains integer in range [0, number of classes - 1]
        
        raise Exception("You have not override getTest() method in Dataset class that inherits from DatasetTemplate")
        return (x_test, y_test)
    
    #Don't override this method
    def getModel(self):
        loadModel(self.param)
        
        #Check if the model can support this dataset
        if self.param.model in self.supported_models:
            return loadModel(self.param)
        else:
            raise Exception("Model '{}' does not support this dataset!".format(self.param.model))