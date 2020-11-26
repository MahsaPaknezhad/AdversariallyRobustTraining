#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility class to handle loading of different model architectures. All models are defined in /models/

from torchvision.models import resnet18, resnet34, resnet50
from models.BasicModel import BasicModel
from models.ResNet import ResNet9, ResNet9Mod, ResNet18, ResNet34, ResNet50
from models.XResNet import XResNet18, XResNet18Mod, XResNet34, XResNet50
from models.WideResNetOriginal import WideResNetOriginal
from models.WideResNetModified import WideResNetModified

def loadModel(param):
    if param.model == 'BasicModel':
        return BasicModel(param).to(param.device)
    elif param.model == 'ResNet9':
        return ResNet9(param).to(param.device)
    elif param.model == 'ResNet9Mod':
        return ResNet9Mod(param.activation).to(param.device)
    elif param.model == 'ResNet18':
        return ResNet18(param.activation).to(param.device)
    elif param.model == 'TorchResNet50':
        return resnet50(pretrained=False, progress=True).to(param.device)
    elif param.model == 'TorchResNet34':
        return resnet34(pretrained=False, progress=True).to(param.device)
    elif param.model == 'TorchResNet18':
        return resnet18(pretrained=False, progress=True).to(param.device)
    elif param.model == 'XResNet18':
        return XResNet18(se = True, params = param, version='3', act_fn = param.activation).to(param.device)
    elif param.model == 'XResNet18Mod':
        return XResNet18_mod(se = True, version='3', act_fn = param.activation).to(param.device)
    elif param.model == 'XResNet34':
        return XResNet34(se = True, version='3', act_fn = param.activation).to(param.device)
    elif param.model == 'XResNet50':
        return XResNet50(se = True, version='3', act_fn = param.activation).to(param.device)
    elif param.model == 'WideResNetModified':
        return WideResNetModified(param.activation, 28, 10, dropout_rate=0.0, num_classes=10).to(param.device)
    elif param.model[:3] == 'WideResNetOriginal':
        model_name = param.model.split('_')
        depth = int(model_name[1])
        multiplier = int(model_name[2])
        return WideResNetOriginal(depth=depth, widen_factor=multiplier, dropout_rate=0, num_classes=10).to(param.device)
    else:
        raise Exception("Model '{}' not implemented".format(param.model))
