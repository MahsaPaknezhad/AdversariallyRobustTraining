#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch.nn as nn
from .Activations import SwishJit, MishJit
from .Downsample import Downsample

class noop(nn.Module):
    def __init__(self):
        super(noop, self).__init__()
        
    def forward(self, x):
        return x

class SE(nn.Module):
    def __init__(self, C, r, act_fn):
        super(SE, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(C, int(C/r))
        self.relu = act(act_fn)
        self.fc2 = nn.Linear(int(C/r), C)
        self.sigmoid = nn.Sigmoid()
        self.C = C

    def forward(self, x):
        mult = self.pool(x).view(x.size(0), -1)
        mult = self.relu(self.fc1(mult))
        mult = self.sigmoid(self.fc2(mult)).view(x.size(0), self.C, 1, 1)

        return mult * x

def act(act_fn):
    if act_fn == 'relu':
        return nn.ReLU(inplace = True)
    elif act_fn == 'none':
        return noop()
    elif act_fn =='mish':
        return MishJit()
    elif act_fn =='swish':
        return SwishJit()

def selfattention(out_planes = 0, se = False, sa = False, sym = False, act_fn = 'relu'):
    if sa:
        return SSA(out_planes, sym = sym)
    elif se:
        return SE(out_planes, r=16, act_fn=act_fn)
    else:
        return noop()

def conv(in_planes, out_planes, ks=3, stride=1, blur=True, num_group = 1):
    if stride == 1 or blur == False:
        return nn.Conv2d(in_planes, out_planes, kernel_size=ks, stride=stride, padding=ks//2, bias=False, groups=num_group)
    else: #stride must be 2 to make sense
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=ks, stride=1, padding=ks//2, bias=False, groups=num_group),
            Downsample(channels=out_planes)
        )

def bn(out_planes, zero_bn = False):
    batchnorm = nn.InstanceNorm2d(out_planes, affine=True, track_running_stats=False)
    nn.init.constant_(batchnorm.weight, 0. if zero_bn else 1.)
    return batchnorm

def conv_2d(in_planes, out_planes, ks=3, stride=1, blur = True, zero_bn=False, act_fn='relu', num_group = 1):
    c = conv(in_planes, out_planes, ks, stride, blur = blur, num_group = num_group)
    b = bn(out_planes, zero_bn)
    a = act(act_fn)
    return nn.Sequential(c, b, a)

def conv_2d_v2(in_planes, out_planes, ks=3, stride=1, blur = True, zero_bn=False, act_fn='relu'):
    b = bn(in_planes, zero_bn)
    a = act(act_fn)
    c = conv(in_planes, out_planes, ks, stride, blur)
    return nn.Sequential(b, a, c)
