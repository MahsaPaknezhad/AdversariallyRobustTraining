#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility class that implements variants of the ResNet.

# Reference:
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#     Deep Residual Learning for Image Recognition. arXiv:1512.03385

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(1, os.path.realpath(os.path.pardir))
from NoiseGenerator import NeighborGenerator, UnlabeledGenerator

class ResNet9(nn.Module):
    def __init__(self, params):
        super(ResNet9, self).__init__()
        
        

        self.params = params

        if 'inject_noise' in self.params:
            if self.params.inject_noise:
                self.unlabeled_generator = UnlabeledGenerator(self.params.unlabeled_noise_std, 1)
                self.neighbor_generator = NeighborGenerator(self.params.neighbor_noise_std, 1)
        self.extract_intermediate_outputs = hasattr(params, 'extract_intermediate_outputs')

        if self.params.activation == 'relu':
            self.activation = nn.ReLU
        elif self.params.activation == 'sigmoid':
            self.activation = nn.Sigmoid
        elif self.params.activation == 'celu':
            self.activation = nn.CELU
    
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride = 1, padding = 1),
            #nn.BatchNorm2d(32),
            nn.InstanceNorm2d(32),
            self.activation())
        
        self.block1_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            #nn.BatchNorm2d(64),
            nn.InstanceNorm2d(64),
            self.activation(),
            nn.MaxPool2d(2, 2))
        
        self.block1_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            #nn.BatchNorm2d(64),
            nn.InstanceNorm2d(64),
            self.activation(),
            nn.Conv2d(64, 64, 3, 1, 1),
            #nn.BatchNorm2d(64),
            nn.InstanceNorm2d(64),
            self.activation())
    
        self.block2_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            #nn.BatchNorm2d(128),
            nn.InstanceNorm2d(128),
            self.activation(),
            nn.MaxPool2d(2, 2))
        
        self.block2_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            #nn.BatchNorm2d(128),
            nn.InstanceNorm2d(128),
            self.activation(),
            nn.Conv2d(128, 128, 3, 1, 1),
            #nn.BatchNorm2d(128),
            nn.InstanceNorm2d(128),
            self.activation())
        
        self.block3_1 = nn.Sequential(
            nn.Conv2d(128, 512, 3, 1, 1),
            #nn.BatchNorm2d(512),
            nn.InstanceNorm2d(512),
            self.activation())
        
        self.block3_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            #nn.BatchNorm2d(512),
            nn.InstanceNorm2d(512),
            self.activation(),
            nn.Conv2d(512, 512, 3, 1, 1),
            #nn.BatchNorm2d(512),
            nn.InstanceNorm2d(512),
            self.activation())

        self.fc = nn.Sequential(
            nn.Linear(512, 10)
        )
   
    def forward(self, x, unlabeled_mode = False):
        #import pdb
        x = self.prep(x)
        x = self.block1_1(x)
        x = x + self.block1_2(x)
        x = self.block2_1(x)
        x = x + self.block2_2(x)
        x = self.block3_1(x)
        #pdb.set_trace()
        x = x + self.block3_2(x)
        x = F.max_pool2d(x, 8, 8)
        # If it is specified in the params that we will inject noise in the intermediate layer, then we shall proceed to do so
        if 'inject_noise' in self.params:
            if self.training and self.params.inject_noise:
                if unlabeled_mode:
                    x[0], _ = self.unlabeled_generator.addUnlabeled(x[0]) 
                original = x[0].reshape([1, 512, 1, 1]).clone()
                if not self.params.jacobian:
                    neighbor = self.neighbor_generator.addNeighbor(x[1]).reshape([1, 512, 1, 1])
                    x = torch.cat((x[0].reshape([1, 512, 1, 1]), neighbor), dim = 0)
            elif not self.training and self.extract_intermediate_outputs:
                return x
        
        x = x.view(-1, 512)
        x = self.fc(x)
        if 'inject_noise' in self.params:
            if self.training and self.params.inject_noise and not self.params.jacobian: return x, original, neighbor
            else: return x
        else: return x

class ResNet9Mod(nn.Module):
    def __init__(self, activation):
        super(ResNet9Mod, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'sigmoid':
            self.activation == nn.Sigmoid
        elif activation == 'celu':
            self.activation = nn.CELU

        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.InstanceNorm2d(32),
            self.activation())

        self.block1_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.InstanceNorm2d(64),
            self.activation(),
            nn.MaxPool2d(2, 2))

        self.block1_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.InstanceNorm2d(64),
            self.activation(),
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.InstanceNorm2d(64),
            self.activation())

        self.block2_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.InstanceNorm2d(128),
            self.activation(),
            nn.MaxPool2d(2, 2))

        self.block2_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.InstanceNorm2d(128),
            self.activation(),
            nn.Conv2d(128, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.InstanceNorm2d(128),
            self.activation())

        self.block3_1 = nn.Sequential(
            nn.Conv2d(128, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.InstanceNorm2d(512),
            self.activation())

        self.block3_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.InstanceNorm2d(512),
            self.activation(),
            nn.Conv2d(512, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.InstanceNorm2d(512),
            self.activation())

        self.fc = nn.Sequential(
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.block1_1(x)
        x = x + self.block1_2(x)
        x = self.block2_1(x)
        x = x + self.block2_2(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        #x = x.view(x.size(0), -1)
        x = F.max_pool2d(x, 8, 8)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

def ResNet18(activation):
    return ResNet(BasicBlock, [2,2,2,2], activation)

def ResNet34(activation):
    return ResNet(BasicBlock, [3,4,6,3], activation)

def ResNet50(activation):
    return ResNet(Bottleneck, [3,4,6,3], activation)

def ResNet101(activation):
    return ResNet(Bottleneck, [3,4,23,3], activation)

def ResNet152(activation):
    return ResNet(Bottleneck, [3,8,36,3], activation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, activation, stride=1):
        super(BasicBlock, self).__init__()
        
        self.activation = activation
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, activation, num_classes=10):
        super(ResNet, self).__init__()
        
        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == 'celu':
            self.activation = nn.CELU(inplace=True)
        
        self.lambda_val = 1
        
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.activation, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def set_lambda(self, lambda_val):
        self.lambda_val = lambda_val