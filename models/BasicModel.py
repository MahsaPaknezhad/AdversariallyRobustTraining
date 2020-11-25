#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility class that implements the basic model used for the MNIST dataset in this paper.

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModel(nn.Module): #LeNet
    def __init__(self, activation):
        super(BasicModel, self).__init__()

        # input channel = 1, output channel = 6, kernel_size = 5
        # input size = (32, 32), output size = (28, 28)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # input channel = 6, output channel = 16, kernel_size = 5
        # input size = (14, 14), output size = (10, 10)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # input dim = 16*5*5, output dim = 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # input dim = 120, output dim = 84
        self.fc2 = nn.Linear(120, 84)
        # input dim = 84, output dim = 10
        self.fc3 = nn.Linear(84, 10)

        self.activation = activation
        self.activation_funcs = {
            'relu': F.relu,
            'leaky relu': F.leaky_relu,
            'sigmoid': torch.sigmoid,
            'tanh': F.tanh
        }

    def forward(self, x):
        # pool size = 2
        # input size = (28, 28), output size = (14, 14), output channel = 6
        x = self.activation_funcs[self.activation](self.conv1(x))
        x = F.max_pool2d(x, 2)
        # pool size = 2
        # input size = (10, 10), output size = (5, 5), output channel = 16
        x = self.activation_funcs[self.activation](self.conv2(x))
        x = F.max_pool2d(x, 2)
        # flatten as one dimension
        x = x.view(x.size()[0], -1)
        # input dim = 16*5*5, output dim = 120
        x = self.activation_funcs[self.activation](self.fc1(x))
        # input dim = 120, output dim = 84
        x = self.activation_funcs[self.activation](self.fc2(x))
        # input dim = 84, output dim = 10
        x = self.fc3(x)
        return x

    def set_lambda(self, lambda_val):
        self.lambda_val = lambda_val