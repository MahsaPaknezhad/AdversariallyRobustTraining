#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility class that implements the basic model used for the MNIST dataset in this paper.
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(1, os.path.realpath(os.path.pardir))
from NoiseGenerator import NeighborGenerator, UnlabeledGenerator

class BasicModel(nn.Module): #LeNet
    def __init__(self, params):
        super(BasicModel, self).__init__()
        # input channel = 1, output channel = 6, kernel_size = 5
        # input size = (32, 32), output size = (28, 28)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # input channel = 6, output channel = 16, kernel_size = 5
        # input size = (14, 14), output size = (10, 10)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # input dim = 16 * 5 * 5 = 400, output dim = 120
        self.fc1 = nn.Linear(400, 120)
        # input dim = 120, output dim = 84
        self.fc2 = nn.Linear(120, 84)
        # input dim = 84, output dim = 10
        self.fc3 = nn.Linear(84, 10)

        self.params = params
        # Get the activation function to use from the params
        self.activation_funcs = {
            'relu': F.relu,
            'leaky relu': F.leaky_relu,
            'sigmoid': torch.sigmoid,
            'tanh': F.tanh
        }

        if 'inject_noise' in self.params:
            if self.params.inject_noise:
                self.unlabeled_generator = UnlabeledGenerator(self.params.unlabeled_noise_std, 1)
                self.neighbor_generator = NeighborGenerator(self.params.neighbor_noise_std, 1)
        self.extract_intermediate_outputs = hasattr(params, 'extract_intermediate_outputs')

    def forward(self, x, unlabeled_mode=False):
        # (32, 32)
        x = self.conv1(x)
        # (28, 28)
        x = self.activation_funcs[self.params.activation](x)
        # (28, 28)
        x = F.max_pool2d(x, 2)
        # (14, 14)
        x = self.conv2(x)
        # (10, 10)
        x = self.activation_funcs[self.params.activation](x)
        # (10, 10)
        x = F.max_pool2d(x, 2)
        # (5, 5)

        # If it is specified in the params that we will inject noise in the intermediate layer, then we shall proceed to do so
        if 'inject_noise' in self.params:
            if self.training and self.params.inject_noise:
                if unlabeled_mode:
                    x[0], _ = self.unlabeled_generator.addUnlabeled(x[0])
                original = x[0].reshape([1, 16, 5, 5])
                if not self.params.jacobian:
                    neighbor = self.neighbor_generator.addNeighbor(x[1]).reshape([1, 16, 5, 5])
                    x = torch.cat((original, neighbor), dim = 0)
            elif not self.training and self.extract_intermediate_outputs:
                return x

        x = x.view(x.size()[0], -1)
        # (400)

        # (400)
        x = self.fc1(x)
        # (120)
        x = self.activation_funcs[self.params.activation](x)
        # (120)
        x = self.fc2(x)
        # (84)
        x = self.activation_funcs[self.params.activation](x)
        # (84)
        x = self.fc3(x)
        # (10)

        if 'inject_noise' in self.params:
            if self.training and self.params.inject_noise and not self.params.jacobian: return x, original, neighbor
            else: return x
        else: return x

    def set_lambda(self, lambda_val):
        self.lambda_val = lambda_val
