#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility class containing the loss functions described in the paper.

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class CELoss(_Loss):
    """Creates a criterion that does Crossentropy Loss. This is a wrapper function
    which executes crossentropy, keeping in mind that target can be -1 i.e. 
    unlabelled anchor

    Shape:
        - logits: :math:`(N, C)` where :math:`C` is number of class
        - targets: :math:`(N, 1)`
        - Output: scalar, or nan if loss cannot be computed
    """

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = F.cross_entropy

    def _getMask(self, targets):
        return (targets != -1)

    def forward(self, logits, targets):
        mask = self._getMask(targets)
        if mask.any():
            logits, targets = logits[mask], targets[mask]
            return self.loss(logits, targets, reduction='none')
        else:
            return torch.tensor([0.], device = 'cuda' if targets.is_cuda else 'cpu')


class GRLoss(_Loss):
    """Creates a criterion that does Differential Geometry Inspired Gradient Regularization given
    the two numerators :math:`fx1`, `fx2` and the two denominators :math:`x1`, `x2`.

    Shape:
        - fx1: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - fx2: :math:`(N x num_neighbor per anchor, *)` 
            where :math:`*` means, any number of additional dimensions
        - x1: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - x2: :math:`(N x num_neighbor per anchor, *)`
            where :math:`*` means, any number of additional dimensions
        - Output: scalar, or nan if loss cannot be computed
    """

    def __init__(self, norm=2):
        super(GRLoss, self).__init__()
        self.norm = norm
        self.loss = F.mse_loss
        
        if norm != 2:
            raise "Current norm not implemented"
        
        self.num_neighbor = None
        
    def forward(self, fx1, fx2, x1, x2):
        if self.num_neighbor is None:
            self.num_neighbor = x2.shape[0] // x1.shape[0]
        
        if self.num_neighbor != 0:
            bs = x2.shape[0]
            fx1 = torch.repeat_interleave(fx1, self.num_neighbor, dim = 0)
            x1 = torch.repeat_interleave(x1, self.num_neighbor, dim = 0)
    
            difference = (x1 - x2).view(bs, -1)
            denom = torch.norm(difference, p = self.norm, dim = -1, keepdim = True)
            fx1 = fx1 / denom
            fx2 = fx2 / denom
    
            return self.loss(fx1, fx2)
        
        else:
            return torch.tensor([0.], device = 'cuda' if fx1.is_cuda else 'cpu')
