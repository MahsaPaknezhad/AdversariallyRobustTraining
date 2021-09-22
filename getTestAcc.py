#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gets the accuracy of the model on the test split of the dataset. No adversarial attack is performed.

import argparse
import numpy as np
import os
import pandas as pd
import pickle
import torch
import math

from DataHandler import DataHandler
from Seed import getSeed
from Logging import info, success
from tqdm import trange

# ---------------------------------------------------------------------------- #
#                                ARGUMENT PARSER                               #
# ---------------------------------------------------------------------------- #

# Dataset parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10', 'Imagenette'], help='Dataset to use for this experiment')
parser.add_argument('--data_dir', type=str, default='/media/mahsa/Element/Regularization/data', help='Folder where the dataset is located')
parser.add_argument('--output_dir', type=str, default='/media/mahsa/Element/Regularization/', help='Directory to output results to. This will also be the same place where the trained model files are read from')
parser.add_argument('--settings_list', type=str, default='model-cifar-ResNet9-adv', help='Semicolon separated string of settings to consider when performing this experiment')
parser.add_argument('--seed_list', type=str, default='27432', help='Semicolon separated string of seeds to consider when performing this experiment')
parser.add_argument('--imsize', type=int, default=32, help='Image size, set to 32 for MNIST and CIFAR10, set to 128 for Imagenette')
# Model parameters.
parser.add_argument('--model', type=str, default='ResNet9', help='Model type, set to basicmodel for MNIST, resnet9 for CIFAR10 and xresnet18 for Imagenette. Check LoadModel.py for the list of supported models.')
parser.add_argument('--activation', type=str, default='celu', help='Activation function for the model, set to sigmoid for MNIST, celu for CIFAR10 and mish for Imagenette')
parser.add_argument('--epoch', type=int, default=100, help='Specify the epoch of the model file to test')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to do forward pass for measurement of clean accuracy')
# Miscellaneous experiment parameters.
parser.add_argument('--seed', type=int, default=0, help='Seed to run experiment. Ignored if time != -1')
parser.add_argument('--time', type=int, default=-1, help='Seed used to generate actual seed to run experiment. Ignored if -1')
parser.add_argument('--device', type=str, default='cuda')
param = parser.parse_args()

# ---------------------------------------------------------------------------- #
#                                    PARAMS                                    #
# ---------------------------------------------------------------------------- #

# Dataset parameters.
dataset = param.dataset
output_dir = param.output_dir
settings_list = param.settings_list.split(',')
seed_list = [s for s in param.seed_list.split(',')]
# Model parameters.
epoch = param.epoch
batch_size = param.batch_size
# Miscellaneous experiment parameters.
device = param.device

# ---------------------------------------------------------------------------- #
#                           INSTANTIATE DATASET CLASS                          #
# ---------------------------------------------------------------------------- #

if dataset == 'MNIST':
    from DatasetMNIST import DatasetMNIST as Dataset
elif dataset == 'CIFAR10':
    from DatasetCIFAR10 import DatasetCIFAR10 as Dataset
elif dataset == 'Imagenette':
    from DatasetImagenette import DatasetImagenette as Dataset

dataset_class = Dataset(param)

x_test_array, y_test_array = dataset_class.getTest()
datahandler = DataHandler(dataset_class, device)
x_test_tensor, y_test_tensor = datahandler.loadTest(x_test_array, y_test_array)
test_size = len(x_test_tensor)
x_test_tensor = x_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

model = dataset_class.getModel()

# ---------------------------------------------------------------------------- #
#                            APPLY TEST SET TO MODEL                           #
# ---------------------------------------------------------------------------- #

results_df = pd.DataFrame(columns=['Setting', 'Seed', 'Clean Test Accuracy'])

for setting in settings_list:

    info(f'Processing {setting}')

    for seed_string in seed_list:

        info(f'Processing {setting}/{seed_string}')

        # Load the model
        '''model_path = os.path.join(target_path, setting, seed_string, f'model_epoch={epoch}.pt')
        model.load_state_dict(torch.load(model_path)['model_state_dict'])'''
        target_path = os.path.join(output_dir, setting, seed_string)
        model_path = os.path.join(target_path, f'model-res-epoch{epoch}.pt')
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval().to(device)

        # Perform the calculation of clean accuracy
        y_output = torch.tensor([], device=device, dtype=torch.long)
        for batchIdx in trange(math.ceil(test_size / batch_size)):
            start_batch = batchIdx * batch_size
            end_batch = min((batchIdx+1)* batch_size, test_size)

            x_batch = x_test_tensor[start_batch:end_batch].to(device)
            y_batch = y_test_tensor[start_batch:end_batch].to(device)

            y_output_batch = model(x_batch).softmax(-1).argmax(-1)
            y_output = torch.cat((y_output, y_output_batch))
        correct_index = torch.where(y_output == y_test_tensor)[0].cpu().numpy()
        clean_accuracy = correct_index.shape[0] / x_test_tensor.shape[0]

        results_df = results_df.append({'Setting': setting, 'Seed': int(seed_string[-5:]), 'Clean Test Accuracy': clean_accuracy}, ignore_index=True)

        success(f'Clean Test Accuracy: {clean_accuracy}')

success(f'Full Results\n{results_df}')

success(f'Aggregated Results\n{results_df.groupby(["Setting"]).mean().reset_index()}')
