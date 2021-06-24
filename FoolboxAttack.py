#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:43:53 2020

@author: ngopc
"""
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

import math 
import json

#from FGSM import fgsm
import foolbox.attacks as fa
from foolbox import PyTorchModel, accuracy, samples
from tqdm import trange
import psutil
import matplotlib.pyplot as plt
import time
import os

class FoolboxAttack:
    def __init__(self, batch_size, device, mean, std, attack_method = 'fgsm'):
        self.batch_size = batch_size
        self.device = device
        
        self.mean = mean
        self.std = std

        self.preprocessing = dict(mean=self.mean, std=self.std, axis=-3)
        self.attack = self._get_attack(attack_method)

    def FoolboxModel(self, model_class, model_name, model_path):
        path = os.path.join(model_path, model_name)
        
        try:
            model_class.load_state_dict(torch.load(path))
        except:
            checkpoint = torch.load(path)
            model_class.load_state_dict(checkpoint['model_state_dict'])
        
        model_class.eval().to(self.device)

        fmodel = PyTorchModel(model_class, bounds=(0, 1), device = self.device, preprocessing=self.preprocessing)
        return fmodel

    def FoolboxModel2(self, model):

        model.eval().to(self.device)
        fmodel = PyTorchModel(model, bounds=(0, 1), device=self.device, preprocessing=self.preprocessing)
        return fmodel

    def getClean(self, fmodel, x_test, y_test):
        device = self.device
        y_output = torch.tensor([], device = device, dtype = torch.long)

        batch_size = self.batch_size
        test_size = x_test.shape[0]
        for batchIdx in range(math.ceil(test_size / batch_size)):
            start_batch = batchIdx * batch_size
            end_batch = min((batchIdx+1)* batch_size, test_size)
            
            x_batch = x_test[start_batch:end_batch].to(device)
            y_batch = y_test[start_batch:end_batch].to(device)

            y_output_batch = fmodel(x_batch).softmax(-1).argmax(-1)
            y_output = torch.cat((y_output, y_output_batch))

        correct_index = torch.where(y_output == y_test)[0].cpu().numpy()
        clean_accuracy = correct_index.shape[0] / x_test.shape[0]

        return clean_accuracy, correct_index.tolist()

    def getRobust(self, fmodel, x_test, y_test, epsilons):
        attack = self.attack
        batch_size = self.batch_size

        rho_array = []
        device = self.device
        test_size = x_test.shape[0]

        successAttack = np.zeros((len(epsilons,)))
        for batchIdx in range(math.ceil(test_size / batch_size)):
            start_batch = batchIdx * batch_size
            end_batch = min((batchIdx+1)* batch_size, test_size)

            x_batch = x_test[start_batch:end_batch].to(device)
            y_batch = y_test[start_batch:end_batch].to(device)
            cur_size = end_batch - start_batch

            _, _, success = attack(fmodel, x_batch, y_batch, epsilons=epsilons)
            success = success.cpu().numpy()
            
            assert success.shape == (len(epsilons), len(x_batch))
            assert success.dtype == np.bool
            successAttack += np.sum(success, axis = -1)
        robust_accuracy = 1 - (successAttack / test_size)
        return robust_accuracy

    def getRho(self, fmodel, x_test, y_test, epsilons):
        attack = self.attack
        batch_size = self.batch_size

        rho_array = []
        success_array = []
        device = self.device
        test_size = x_test.shape[0]

        successAttack = 0
        for batchIdx in trange(math.ceil(test_size / batch_size)):
            start_batch = batchIdx * batch_size
            end_batch = min((batchIdx + 1) * batch_size, test_size)

            x_batch = x_test[start_batch:end_batch].to(device)
            y_batch = y_test[start_batch:end_batch].to(device)
            cur_size = end_batch - start_batch

            _, advs, success = attack(fmodel, x_batch, y_batch, epsilons=[None])
            numer = torch.norm(advs[0] - x_batch, p=2, dim=(1, 2, 3))
            denom = torch.norm(x_batch, p=2, dim=(1, 2, 3))
            rho = numer / denom
            success_array.append(torch.squeeze(success, 0))
            success = success.cpu().numpy()

            assert success.shape == (1, len(x_batch))
            assert success.dtype == np.bool
            successAttack += np.sum(success, axis=-1)
            rho_array.append(rho)
        rho_tensor = torch.cat(rho_array, dim=0)
        success_tensor = torch.cat(success_array, dim=0)
        rho_mean = torch.mean(rho_tensor[success_tensor], dim=0)
        robust_accuracy = 1 - (successAttack / test_size)
        return robust_accuracy, rho_mean.cpu().numpy()

    def getAdv(self, fmodel, x_test, y_test, epsilons):
        attack = self.attack

        successAttack = np.zeros((len(epsilons, )))
        _, advs, success = attack(fmodel, x_test, y_test, epsilons=epsilons)
        success = success.cpu().numpy()
        assert success.shape == (len(epsilons), len(x_test))
        assert success.dtype == np.bool
        return advs, success

    def _get_attack(self, attack):
        if attack == 'fgsm':
            return fa.FGSM()
        elif attack == 'pgd':
            return fa.PGD()
        elif attack == 'deepfooll2':
            return fa.LinfDeepFoolAttack()
        elif attack == 'deepfoollinf':
            return fa.L2DeepFoolAttack()
        else:
            raise Exception("Attack not implemented")

class AttackLogging():
    def __init__(self, numEpsilons):
        self.correct_index = []
        self.clean = {}
        self.robust = np.zeros((1, numEpsilons))
        self.rho = np.zeros((1, numEpsilons))
    
    def logClean(self, setting_name, clean_accuracy, correct_index):
        self._collectIdx(correct_index)
        self._collectClean(setting_name, clean_accuracy)
    
    def _collectIdx(self, correct_index):
        if len(self.correct_index) == 0:
            self.correct_index = correct_index
        else:
            correctIndex = set(self.correct_index)
            tempIndex = set(correct_index)
            self.correct_index = list(correctIndex.intersection(tempIndex))
    
    def _collectClean(self, setting_name, clean_accuracy):
        self.clean[setting_name] = clean_accuracy

    def logRobust(self, setting_name, robust_accuracy):
        self.robust = np.concatenate((self.robust, np.expand_dims(robust_accuracy, 0)), axis = 0)

    def logRho(self, setting_name, rho):
        self.rho = np.concatenate((self.rho, np.expand_dims(rho, 0)), axis = 0)
    
    def saveLogs(self, seed, path):
        with open(os.path.join(path,'cleanAccuracy_seed={}.json'.format(seed)), 'w') as file:
            json.dump(self.clean, file, indent=4)
        
        np.save(os.path.join(path,'robustAccuracy_seed={}.npy'.format(seed)), self.robust)



    
# class FoolboxAttack():
#     def __init__(self, model_class, model_names, x_test, y_test, param, mean, std):
#         self.param = param

#         self.model_class = model_class
#         self.model_names = model_names

#         self.x_test = x_test.to(self.param.device)
#         self.y_test = y_test.to(self.param.device)

#         self.mean = mean
#         self.std = std
#         self.attack = self._get_attack(param.attack_method)
#         self.epsilons = self._getEpsilons(param.min_epsilon, param.max_epsilon, param.step_epsilon)

#         self.numEpsilons = len(self.epsilons)
#         self.numModels = len(self.model_names)

#         # Logging purposes
#         self.correct_index = []
#         self.clean = {}
#         self.robust = np.zeros((1, self.numEpsilons))

#         assert self.x_test.is_cuda == (self.param.device == 'cuda')
#         assert self.y_test.is_cuda == (self.param.device == 'cuda')


#     def FoolboxModel(self, model_name):
#         model = self._TorchModel(model_name)
#         preprocessing = dict(mean=self.mean, std=self.std, axis=-3) # Need to adapt to MNIST

#         fmodel = PyTorchModel(model, bounds=(0, 1), device = self.param.device, preprocessing=preprocessing)
#         return fmodel
    
#     def getClean(self, fmodel):
#         device = self.param.device
#         batch_size = self.param.batch_size

#         y_output = torch.tensor([], device = device, dtype = torch.long)

#         test_size = self.x_test.shape[0]
#         for batchIdx in range(math.ceil(test_size / batch_size )):
#             start_batch = batchIdx * batch_size
#             end_batch = min((batchIdx+1)* batch_size, test_size)
            
#             x_batch = self.x_test[start_batch:end_batch].to(device)
#             y_batch = self.y_test[start_batch:end_batch].to(device)

#             y_output_batch = fmodel(x_batch).softmax(-1).argmax(-1)
#             y_output = torch.cat((y_output, y_output_batch))

#         correct_index = torch.where(y_output == self.y_test)[0].cpu().numpy()
#         clean_accuracy = correct_index.shape[0] / self.x_test.shape[0]

#         return clean_accuracy, correct_index.tolist()
    
#     def logClean(self, model_name, clean_accuracy, correct_index):
#         self._collectIdx(correct_index)
#         self._collectClean(model_name, clean_accuracy)

#     def getRobust(self, fmodel):
#         assert len(self.correct_index) != 0
#         epsilons = self.epsilons

#         device = self.param.device
#         batch_size = self.param.batch_size

#         successAttack = np.zeros((len(epsilons,)))

#         x_test, y_test = self._getCommonDataset()
#         test_size = x_test.shape[0]

#         for batchIdx in range(math.ceil(test_size / batch_size )):
#             start_batch = batchIdx * batch_size
#             end_batch = min((batchIdx+1)* batch_size, test_size)

#             x_batch = self.x_test[start_batch:end_batch].to(device)
#             y_batch = self.y_test[start_batch:end_batch].to(device)

#             _, _, success = self.attack(fmodel, x_batch, y_batch, epsilons=epsilons)
#             success = success.cpu().numpy()
            
#             assert success.shape == (len(epsilons), len(x_batch))
#             assert success.dtype == np.bool
#             successAttack += np.sum(success, axis = -1)

#         robust_accuracy = 1 - (successAttack / test_size)
#         return robust_accuracy
    
#     def logRobust(self, model_name, robust_accuracy):
#         self.robust = np.concatenate((self.robust, np.expand_dims(robust_accuracy, 0)), axis = 0)       
    
#     def saveLogs(self, seed):
#         path = self.param.model_dir

#         with open(os.path.join(path,'cleanAccuracy_seed={}.json'.format(seed)), 'w') as file:
#             json.dump(self.clean, file, indent=4)

#         np.save(os.path.join(path,'robustAccuracy_seed={}.npy'.format(seed)), self.robust)

#     def _get_attack(self, attack):
#         if attack == 'fgsm':
#             return fa.FGSM()
#         elif attack == 'pgd':
#             return fa.PGD()
#         else:
#             raise Exception("Attack not implemented")
    
#     def _TorchModel(self, model_name):
#         path = os.path.join(self.param.model_dir, model_name)
#         model = self.model_class
#         model.load_state_dict(torch.load(path))
#         return model.eval().to(self.param.device)

#     def _collectIdx(self, correct_index):
#         if len(self.correct_index) == 0:
#             self.correct_index = correct_index
#         else:
#             correctIndex = set(self.correct_index)
#             tempIndex = set(correct_index)
#             self.correct_index = list(correctIndex.intersection(tempIndex))
    
#     def _collectClean(self, model_name, clean_accuracy):
#         self.clean[model_name] = clean_accuracy
    
#     def _getCommonDataset(self):
#         correctIndex = np.array(self.correct_index)
#         return self.x_test[correctIndex], self.y_test[correctIndex]

#     def _getEpsilons(self, minEpsilon, maxEpsilon, step):
#         return np.arange(minEpsilon, maxEpsilon, step)
