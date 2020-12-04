#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility class that implements XResNet variants.
import torch
import torch.nn as nn
from ImagenetteUtils.Downsample import Downsample
from ImagenetteUtils.Operations import conv, bn, act, conv_2d, selfattention, conv_2d_v2
import sys
import os
sys.path.insert(1, os.path.realpath(os.path.pardir))
from NoiseGenerator import NeighborGenerator, UnlabeledGenerator

# Include support for resnext

def XResNet18(version = '1', **kwargs):
    # Constructs a XResNet-18 model.
    layers = [2, 2, 2, 2]

    if version == '1':
        model = XResNet(BasicBlock_v1, layers, **kwargs)
    elif version == '2':
        model = XResNet(BasicBlock_v2, layers, **kwargs)
    elif version == '3':
        model = XResNet(BasicBlock_v3, layers, **kwargs)
    return model

def XResNet18Mod(version='1', **kwargs):
    # Constructs a modified variant of XResNet18 model.
    layers = [2, 2, 2, 2]
    if version == '1':
        model = XResNetMod(BasicBlock_v1, layers, **kwargs)
    elif version == '2':
        model = XResNetMod(BasicBlock_v2, layers, **kwargs)
    elif version == '3':
        model = XResNetMod(BasicBlock_v3, layers, **kwargs)
    return model

def XResNet34(version='1', **kwargs):
    # Constructs a XResNet-34 model.
    layers = [3, 4, 6, 3]
    if version == '1':
        model = XResNet(BasicBlock_v1, layers, **kwargs)
    elif version == '2':
        model = XResNet(BasicBlock_v2, layers, **kwargs)
    elif version == '3':
        model = XResNet(BasicBlock_v3, layers, **kwargs)
    return model

def XResNet50(version='1', **kwargs):
    # Constructs a XResNet-50 model.
    layers = [3, 4, 6, 3]
    if version == '1':
        model = XResNet(Bottleneck_v1, layers, **kwargs)
    elif version == '2':
        model = XResNet(Bottleneck_v2, layers, **kwargs)
    elif version == '3':
        model = XResNet(Bottleneck_v3, layers, **kwargs)
    return model

def XResNet101(version='1', **kwargs):
    # Constructs a XResNet-101 model.
    layers = [3, 4, 23, 3]
    if version == '1':
        model = XResNet(Bottleneck_v1, layers, **kwargs)
    elif version == '2':
        model = XResNet(Bottleneck_v2, layers, **kwargs)
    elif version == '3':
        model = XResNet(Bottleneck_v3, layers, **kwargs)
    return model

def XResNet152(version='1', **kwargs):
    # Constructs a XResNet-152 model.
    layers = [3, 8, 36, 3]
    if version == '1':
        model = XResNet(Bottleneck_v1, layers, **kwargs)
    elif version == '2':
        model = XResNet(Bottleneck_v2, layers, **kwargs)
    elif version == '3':
        model = XResNet(Bottleneck_v3, layers, **kwargs)
    return model

class XResNet(nn.Module):
    def __init__(self, block, layers, params, c_out=10, drop_prob = 0.2, **kwargs):
        self.inplanes = 64
        super(XResNet, self).__init__()
        self.conv1 = conv_2d(3, 32, stride=2, blur=False)
        self.conv2 = conv_2d(32, 32, stride=1)
        self.conv3 = conv_2d(32, 64, stride=1)
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Downsample(channels=64)
        )
        self.layer1 = self._make_layer(block, 64, layers[0], **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(512 * block.expansion, c_out)
        )
    
        self.params = params
        if 'inject_noise' in self.params:
            if self.params.inject_noise:
                self.unlabeled_generator = UnlabeledGenerator(self.params.unlabeled_noise_std, 1)
                self.neighbor_generator = NeighborGenerator(self.params.neighbor_noise_std, 1)
        self.extract_intermediate_outputs = hasattr(params, 'extract_intermediate_outputs')
    
    def forward(self, x, unlabeled_mode = False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)

        #ResNet Blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if 'inject_noise' in self.params:
            if self.training and self.params.inject_noise:
                if unlabeled_mode:
                    x[0], _ = self.unlabeled_generator.addUnlabeled(x[0])
                original = x[0].reshape([1, 16, 5, 5])
                if not self.params.jacobian:
                    neighbor = self.neighbor_generator.addNeighbor(x[1]).reshape([1, 16, 5, 5])
                    x = torch.cat((original, neighbor), dim = 0)
        if not self.training and 'extract_intermediate_outputs' in self.params:
            return x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if 'inject_noise' in self.params:
            if self.training and self.params.inject_noise and not self.params.jacobian: return x, original, neighbor
            else: return x
        else: return x

    def _make_ds(self, block, planes, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            out = planes * block.expansion

            if stride==2:
                ds = Downsample(channels=self.inplanes)
                layers.append(ds)

            c = conv(self.inplanes, out, ks=1, stride=1)
            b = bn(out)
            layers.append(c)
            layers.append(b)
            downsample = nn.Sequential(*layers)
        return downsample

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs): # Generate Large Block using the smaller bottleneck/basic blocks
        layers = []
        downsample = self._make_ds(block, planes, stride)
        layers.append(block(self.inplanes, planes, stride, downsample, **kwargs))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, **kwargs))

        return nn.Sequential(*layers)

class XResNetMod(nn.Module):
    def __init__(self, block, layers, c_out=10, drop_prob=0.2, **kwargs):
        self.inplanes = 64
        super(XResNetMod, self).__init__()
        self.conv1 = conv_2d(3, 32, stride=2, blur=False)
        self.conv2 = conv_2d(32, 32, stride=1)
        self.conv3 = conv_2d(32, 64, stride=1)
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Downsample(channels=64))
        self.layer1 = self._make_layer(block, 64, layers[0], **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)

        # ResNet Blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return x


    def _make_ds(self, block, planes, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            out = planes * block.expansion

            if stride == 2:
                ds = Downsample(channels=self.inplanes)
                layers.append(ds)

            c = conv(self.inplanes, out, ks=1, stride=1)
            b = bn(out)
            layers.append(c)
            layers.append(b)
            downsample = nn.Sequential(*layers)
        return downsample

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):  # Generate Large Block using the smaller bottleneck/basic blocks
        layers = []
        downsample = self._make_ds(block, planes, stride)
        layers.append(block(self.inplanes, planes, stride, downsample, **kwargs))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, **kwargs))

        return nn.Sequential(*layers)

class BasicBlock_v1(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, act_fn = 'relu', se = False, sa=False, sym=False):
        super(BasicBlock_v1, self).__init__()
        self.conv1 = conv_2d(inplanes, planes, stride=stride, act_fn=act_fn)
        self.conv2 = conv_2d(planes, planes, act_fn = 'none', zero_bn = sa)
        self.stride = stride
        self.relu = act(act_fn)
        self.ssa = selfattention(planes, se, sa, sym, act_fn)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.ssa(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class BasicBlock_v2(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, act_fn = 'relu', se = False, sa=False, sym=False):
        super(BasicBlock_v2, self).__init__()
        self.conv1 = conv_2d_v2(inplanes, planes, stride=stride, act_fn=act_fn)
        self.conv2 = conv_2d_v2(planes, planes, act_fn = 'none', zero_bn = sa)
        self.stride = stride
        self.relu = act(act_fn)
        self.ssa = selfattention(planes, se, sa, sym, act_fn)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.ssa(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class BasicBlock_v3(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, act_fn = 'relu', se = False, sa=False, sym=False, num_group=32):
        super(BasicBlock_v3, self).__init__()
        self.conv1 = conv_2d(inplanes, planes * 2, stride=stride, act_fn=act_fn)
        self.conv2 = conv_2d(planes * 2, planes, act_fn = 'none', zero_bn = sa, num_group = num_group)
        self.stride = stride
        self.relu = act(act_fn)
        self.ssa = selfattention(planes, se, sa, sym, act_fn)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.ssa(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class Bottleneck_v1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_fn = 'relu', se = False, sa = False, sym = False):
        super(Bottleneck_v1, self).__init__()
        self.conv1 = conv_2d(inplanes, planes, ks = 1, act_fn = act_fn)
        self.conv2 = conv_2d(planes, planes, 3, stride, act_fn = act_fn)
        self.conv3 = conv_2d(planes, planes * self.expansion, 3, 1, zero_bn = sa, act_fn = 'none')
        self.relu = act(act_fn)
        self.stride = stride
        self.ssa = selfattention(planes * self.expansion, se, sa, sym, act_fn)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.ssa(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_v2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_fn = 'relu', se = False, sa = False, sym = False):
        super(Bottleneck_v2, self).__init__()
        self.conv1 = conv_2d_v2(inplanes, planes, ks = 1, act_fn = act_fn)
        self.conv2 = conv_2d_v2(planes, planes, 3, stride, act_fn = act_fn)
        self.conv3 = conv_2d_v2(planes, planes * self.expansion, 3, 1, zero_bn = sa, act_fn = 'none')
        self.relu = act(act_fn)
        self.stride = stride
        self.ssa = selfattention(planes * self.expansion, se, sa, sym, act_fn)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.ssa(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class Bottleneck_v3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_fn = 'relu', se = False, sa = False, sym = False, num_group=32):
        super(Bottleneck_v3, self).__init__()
        self.conv1 = conv_2d(inplanes, planes * 2, ks = 1, act_fn = act_fn)
        self.conv2 = conv_2d(planes * 2, planes * 2, 3, stride, act_fn = act_fn, num_group = num_group)
        self.conv3 = conv_2d(planes * 2, planes * self.expansion, 3, 1, zero_bn = sa, act_fn = 'none')
        self.relu = act(act_fn)
        self.stride = stride
        self.ssa = selfattention(planes * self.expansion, se, sa, sym, act_fn)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.ssa(self.conv3(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out