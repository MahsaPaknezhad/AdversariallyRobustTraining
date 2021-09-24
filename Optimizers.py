#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility file that contains some optimizers for training.

import torch
# This is the optimizer class built into pytorch
from torch import optim as optim1
# This is the third party optimizer class that came from pip install torch_optimizer
#import torch_optimizer as optim2

class Optimizers:
    def Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0):
        return optim1.Adadelta(params, lr, rho, eps, weight_decay)
    
    def Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        return optim1.Adagrad(params, lr, lr_decay, weight_decay, initial_accumulator_value, eps)
    
    def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        return optim1.Adam(params, lr, betas, eps, weight_decay, amsgrad)
    
    def AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False):
        return optim1.AdamW(params, lr, betas, eps, weight_decay, amsgrad)
    
    def Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        return optim1.Adamax(params, lr, betas, eps, weight_decay)

    def RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        return optim1.RMSprop(params, lr, alpha, eps, weight_decay, momentum, centered)
    
    def SGD(params, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        return optim1.SGD(params, lr, momentum, dampening, weight_decay, nesterov)
    
#    def AdaBound(params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3, eps=1e-8, weight_decay=0, amsbound=False):
#        return optim2.AdaBound(params, lr, betas, final_lr, gamma, eps, weight_decay, amsbound)
    
#    def Ranger(params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-5, weight_decay=0):
#        return optim2.Ranger(params, lr, alpha, k, N_sma_threshhold, betas, eps, weight_decay)

