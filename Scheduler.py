#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:01:03 2020

@author: amadeusaw
"""

import math
#import warnings
#from torch.optim.lr_scheduler import _LRScheduler

class LinearCosineLR():
    """Sets the learning rate of each parameter group to the initial lr
    for a given amount of time (Constant schedule). Afterwards, execute a
    cosine annealing schedule. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer):          Wrapped optimizer.
        total_steps (int):              Number of updates to be done
        linear_frac (float, optional):  Fraction of total updates using the constant scheduler.
                                        Used only if linear_steps is None
        linear_steps (int, optional):   Number of updates to use the constant scheduler. 
                                        If linear_steps is None, use linear_frac instead. Default: None
        last_epoch (int, optional):     The index of last epoch. Default: -1.

    Example:
        >>> scheduler = LinearCosineLR(optimizer, total_steps = 10)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, total_steps, linear_frac = 0.5, linear_steps=None, last_batch=-1):
        self.optimizer = optimizer

        self.total_steps = total_steps
        self.linear_frac = linear_frac
        
        if linear_steps is None:
            self.linear_steps = math.ceil(total_steps * linear_frac)
        else:
            self.linear_steps = linear_steps
        self.cosine_steps = total_steps - self.linear_steps

        self.base_lr = self.optimizer.param_groups[0]['lr']
        self.last_batch = last_batch
        
    def step(self):
        self.last_batch += 1
        self.optimizer.param_groups[0]['lr'] = self._getLR(self.last_batch)
    
    def _getLR(self, currentIter):
        if currentIter <= self.linear_steps:
            return self.base_lr
        else:
            return self._getCos(currentIter)
    
    def _getCos(self, currentIter):
        actualIter = currentIter - self.linear_steps
        return self.base_lr / 2 * (math.cos(math.pi * actualIter / self.cosine_steps) + 1)