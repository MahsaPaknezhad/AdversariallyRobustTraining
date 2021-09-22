#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:05:41 2020

@author: ngopc
"""
import torch
import torch.optim as optim

class LRScheduler():
    def __init__(self, optimizer, step_size, gamma):
        self.step_size = step_size
        self.gamma = gamma
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)