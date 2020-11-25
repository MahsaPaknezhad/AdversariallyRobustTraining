#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility file that contains some learning rate schedulers.

import torch.optim.lr_scheduler
import math

def StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1):
    return StepLR(optimizer, step_size, gamma, last_epoch)

def CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
    return CyclicLR(optimizer, base_lr, max_lr, step_size_up, step_size_down, mode, gamma, scale_fn, scale_mode, cycle_momentum, base_momentum, max_momentum, last_epoch)

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
        return self.base_lr / 2 * (math.cos(math.pi * actualIter / self.cosine_steps) + 1)#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:04:44 2020

@author: amadeusaw
"""
import torch.optim.lr_scheduler
import math

def StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1):
    return StepLR(optimizer, step_size, gamma, last_epoch)

def CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
    return CyclicLR(optimizer, base_lr, max_lr, step_size_up, step_size_down, mode, gamma, scale_fn, scale_mode, cycle_momentum, base_momentum, max_momentum, last_epoch)

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
        return self.base_lr / 2 * (math.cos(math.pi * actualIter / self.cosine_steps) + 1)#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:04:44 2020

@author: amadeusaw
"""
import torch.optim.lr_scheduler
import math

def StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1):
    return StepLR(optimizer, step_size, gamma, last_epoch)

def CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
    return CyclicLR(optimizer, base_lr, max_lr, step_size_up, step_size_down, mode, gamma, scale_fn, scale_mode, cycle_momentum, base_momentum, max_momentum, last_epoch)

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
        return self.base_lr / 2 * (math.cos(math.pi * actualIter / self.cosine_steps) + 1)#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:04:44 2020

@author: amadeusaw
"""
import torch.optim.lr_scheduler
import math

def StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1):
    return StepLR(optimizer, step_size, gamma, last_epoch)

def CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
    return CyclicLR(optimizer, base_lr, max_lr, step_size_up, step_size_down, mode, gamma, scale_fn, scale_mode, cycle_momentum, base_momentum, max_momentum, last_epoch)

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