#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import argparse
import os
from tqdm import trange
import numpy as np
from DataHandler import DataHandler
from Logging import info, success, warn
import numpy as np
import matplotlib.pyplot as plt
import pdb

# ---------------------------------------------------------------------------- #
#                                ARGUMENT PARSER                               #
# ---------------------------------------------------------------------------- #

parser = argparse.ArgumentParser()
# Dataset parameters.
parser.add_argument('--dataset', type=str, choices=['MNIST', 'CIFAR10', 'Imagenette'], help='Dataset to be used for experiment')
parser.add_argument('--data_dir', type=str, default='../data', help='Directory to download dataset to')
parser.add_argument('--output_dir', type=str, default='../output', help='Directory to dump output image in')
parser.add_argument('--imsize', type=int, help='Image size, set to 32 for MNIST and CIFAR10, set to 128 for Imagenette')
# Model parameters.
parser.add_argument('--model', help='Model type, set to basicmodel for MNIST, resnet9 for CIFAR10 and xresnet18 for Imagenette. Check LoadModel.py for the list of supported models.')
parser.add_argument('--activation', help='Activation function for the model, set to sigmoid for MNIST, celu for CIFAR10 and mish for Imagenette')
parser.add_argument('--pretrained_model_path', type=str, help='Path to the model file that was pretrained on the dataset to load')
parser.add_argument('--num_class', type=int, help='Number of classes')
parser.add_argument('--num_train', type=int, help='Number of samples to generate std from') # Named to maintain compatability with DataHandler
parser.add_argument('--seed', type=int, default=0, help='Seed')
# Miscellaneous experiment parameters
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to do inference in this experiment')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to perform experiment on')
param = parser.parse_args()

# ---------------------------------------------------------------------------- #
#                                    PARAMS & SETTINGS                         #
# ---------------------------------------------------------------------------- #

# Dataset parameters.
dataset = param.dataset
output_dir = param.output_dir

# Model parameters.
pretrained_model_path = param.pretrained_model_path

# Miscellaneous experiment parameters.
batch_size = param.batch_size
device = param.device

# Set the flag to tell the model that we want it to output intermediate layers
param.extract_intermediate_outputs = True

# Insert additional parameters
param.inject_noise = False
param.num_val = 0

# ---------------------------------------------------------------------------- #
#                              INSTANTIATE DATASET                             #
# ---------------------------------------------------------------------------- #

if dataset == 'MNIST':
    from DatasetMNIST import DatasetMNIST as Dataset
elif dataset == 'CIFAR10':
    from DatasetCIFAR10 import DatasetCIFAR10 as Dataset
elif dataset == 'Imagenette':
    from DatasetImagenette import DatasetImagenette as Dataset
else:
    raise Exception('Unsupported dataset.')
    exit
    
dataset_class = Dataset(param)
(x_train_array, y_train_array), _ = dataset_class.getTrainVal(train_val_ratio=1.0)
datahandler = DataHandler(dataset_class, 'cpu')
x_tensor, _ = datahandler.loadAugmentedLabeled(x_train_array, y_train_array)

# ---------------------------------------------------------------------------- #
#                             LOAD PRETRAINED MODEL                            #
# ---------------------------------------------------------------------------- #

model = dataset_class.getModel()

if os.path.isfile(pretrained_model_path):
    checkpoint = torch.load(pretrained_model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    success(f'Successfully loaded model file from {pretrained_model_path}.')
else:
    raise Exception(f'No model file found at {pretrained_model_path}.')
    exit

# ---------------------------------------------------------------------------- #
#                         GENERATE INTERMEDIATE OUTPUTS                        #
# ---------------------------------------------------------------------------- #

info('Generating intermediate outputs.')

intermediate_outputs = torch.Tensor()
with torch.no_grad():
    for batchIdx in trange(math.ceil(x_tensor.shape[0]/batch_size)):
        start_batch = batchIdx*batch_size
        end_batch = min((batchIdx+1)*batch_size, x_tensor.shape[0])

        x_anchor = x_tensor[start_batch:end_batch]
        x_anchor = x_anchor.to(device)

        x_intermediate = model(x_anchor).to(device)
        intermediate_outputs = torch.cat((intermediate_outputs.to(device), x_intermediate), dim=0)

# ---------------------------------------------------------------------------- #
#                         PROCESS INTERMEDIATE OUTPUTS                         #
# ---------------------------------------------------------------------------- #

info('Processing intermediate outputs.')

histogram_values = []

for i in trange(intermediate_outputs.shape[0]):
    for j in range(i+1, intermediate_outputs.shape[0]): # dont mess up your own trange.
        image1 = intermediate_outputs[i]
        image2 = intermediate_outputs[j]
        diff = (image1 - image2)**2
        diff = diff.flatten()
        hist_value = np.sqrt(torch.sum(diff.detach()).item())/(intermediate_outputs.shape[1] * intermediate_outputs.shape[2] * intermediate_outputs.shape[3])
        histogram_values.append(hist_value)

# ---------------------------------------------------------------------------- #
#                           PLOT HISTOGRAM WITH DATA                           #
# ---------------------------------------------------------------------------- #

mu = np.mean(histogram_values)
var = np.var(histogram_values)

rc_params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6, 6),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(rc_params)
histogram_values = np.asarray(histogram_values)
info(f'Mean: {mu}')
info(f'Variance: {var}')
n, bins, patches = plt.hist(x=histogram_values, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('L2 Norm Histogram')
plt.text(0.02, 7000, r'$\mu$=%f, var=%f'%(mu, var))
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()
save_path = os.path.join(output_dir, 'NoiseInjectionHistogram.png')
plt.savefig(save_path)
success(f'Saved graph to {save_path}')