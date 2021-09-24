#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:04:44 2020

@author: amadeusaw
"""
import torch.optim.lr_scheduler as scheduler1
import Scheduler as scheduler2

class lr_scheduler:
    def StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1):
        return scheduler1.StepLR(optimizer, step_size, gamma, last_epoch)
    
    def CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
        return scheduler1.CyclicLR(optimizer, base_lr, max_lr, step_size_up, step_size_down, mode, gamma, scale_fn, scale_mode, cycle_momentum, base_momentum, max_momentum, last_epoch)
    
    def LinearCosineLR(optimizer, total_steps, linear_frac = 0.5, linear_steps=None, last_batch=-1):
        return scheduler2.LinearCosineLR(optimizer, total_steps, linear_frac, linear_steps, last_batch)
    