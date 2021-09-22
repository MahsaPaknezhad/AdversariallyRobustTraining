#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gets the adversarial ratio which is essentially the percentage of the image that needs to be perturbed before the adversarial attack is considered succesful
# The higher this ratio, the more robust the model is
# In this case, the attack used in the paper was DeepFool

import math
import torch
import argparse
import numpy as np
import pandas as pd
import pickle
import os

from DeepFool import deepfool
from tqdm import trange
from DataHandler import DataHandler
from Seed import getSeed
from Logging import info, success

# ---------------------------------------------------------------------------- #
#                                ARGUMENT PARSER                               #
# ---------------------------------------------------------------------------- #

parser = argparse.ArgumentParser()
# Dataset parameters.
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10', 'Imagenette'], help='Dataset to use for this experiment')
parser.add_argument('--data_dir', type=str, default='/media/mahsa/Element/Regularization/data', help='Folder where the dataset is located')
parser.add_argument('--output_dir', type=str, default='/media/mahsa/Element/Regularization/', help='Directory to output results to. This will also be the same place where the trained model files are read from')
parser.add_argument('--settings_list', type=str, default='model-cifar-ResNet9-adv', help='Comma separated string of settings to consider when performing this experiment')
parser.add_argument('--seed_list', type=str, default='27432', help='Comma separated string of seeds to consider when performing this experiment')#27432,30416,48563,51985,842016
parser.add_argument('--imsize', type=int, default= 32, help='Image size, set to 32 for MNIST and CIFAR10, set to 128 for Imagenette')
# Model parameters.
parser.add_argument('--model', type=str, default='ResNet9', help='Model type, set to basicmodel for MNIST, resnet9 for CIFAR10 and xresnet18 for Imagenette. Check LoadModel.py for the list of supported models.')
parser.add_argument('--activation', type=str, default='celu', help='Activation function for the model, set to sigmoid for MNIST, celu for CIFAR10 and mish for Imagenette')
parser.add_argument('--epoch', type=int, default=60, help='Specify the epoch of the model file to test')
# Adversarial attack parameters.
parser.add_argument('--max_iter', type=int, default= 9, help='Max number of iterations that the DeepFool algorithm can run when doing the attack. Set to 23 for MNIST, 9 for CIFAR10 and 4 for Imagenette')
# Miscellaneous experiment parameters
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
# Adversarial attack parameters.
max_iter = param.max_iter
# Miscellaneous experiment parameters.
if param.time != -1:
    param.seed = getSeed(time = param.time)
device = param.device

# ---------------------------------------------------------------------------- #
#                           INSTANTIATE DATASET CLASS                          #
# ---------------------------------------------------------------------------- #

if dataset == "MNIST":
    from DatasetMNIST import DatasetMNIST as Dataset
elif dataset == "CIFAR10":
    from DatasetCIFAR10 import DatasetCIFAR10 as Dataset
elif dataset == "Imagenette":
    from DatasetImagenette import DatasetImagenette as Dataset

dataset_class = Dataset(param)

x_test_array, y_test_array = dataset_class.getTest()
datahandler= DataHandler(dataset_class, 'cpu')
x_test_tensor, y_test_tensor = datahandler.loadTest(x_test_array, y_test_array)
test_size = x_test_tensor.shape[0]

model = dataset_class.getModel()

# ---------------------------------------------------------------------------- #
#                              GET ROBUST RATIO                                #
# ---------------------------------------------------------------------------- #

results_df = pd.DataFrame(columns=['Setting', 'Seed', 'Rho Value', 'Successful Attack Ratio'])

for j, setting in enumerate(settings_list):

    info(f'Processing setting {setting}')

    for seed_string in seed_list:

        info(f'Processing {setting}/{seed_string}')

        # Load model.
        target_path = os.path.join(output_dir, setting, seed_string)
        model_path = os.path.join(target_path, f'model-res-epoch{epoch}.pt')
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        '''model_path = os.path.join(target_path, setting, seed_string, f'model_epoch={epoch}.pt')
        model.load_state_dict(torch.load(model_path)['model_state_dict'])'''
        success(f'Successfully loaded model from {model_path}.')

        # Try to generate DeepFool adversarial images.
        rho_array = []
        success_count = 0
        for i in trange(test_size):
            # Get batch of image data and label data.
            x_sample = torch.squeeze(x_test_tensor[i], 0).to(device)
            if dataset == 'MNIST': x_sample = x_sample.reshape([1, 32, 32])
            y_sample = torch.squeeze(y_test_tensor[i], 0).to(device)

            # Generate DeepFool perturbed image.
            r, loop_i, label_orig, label_pert, pert_image = deepfool(x_sample, model, max_iter=max_iter)

            # The adversarial attack is deemed successful if the label changes.
            if label_orig != label_pert:
                success_count += 1

            # Calcualate rho
            perturbation = torch.squeeze(pert_image, 0)
            numerator = torch.norm(perturbation - x_sample, p=2, dim=(0, 1, 2))
            denominator = torch.norm(x_sample, p=2, dim=(0, 1, 2))
            rho = numerator / denominator

            if not np.isnan(rho.cpu().item()):
                rho_array.append(rho.cpu().item())

        # Calculate mean rho and success ratio and log them into DataFrame.
        rho_mean = np.mean(rho_array)
        success_ratio = success_count / test_size
        
        success(f'Mean Rho {rho_mean}, Successful Attack Ratio {success_ratio}')
        
        results_df = results_df.append({"Setting": setting, "Seed": int(seed_string[-5:]), "Rho Value": rho_mean, "Successful Attack Ratio": success_ratio}, ignore_index=True)

    save_path = os.path.join(output_dir, setting, f"Rho Results Max Iter {max_iter} Checkpoint {j+1}.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(results_df, f)
        success(f'Results saved to {save_path}')

success(f'Full Results\n{results_df}')

success(f'Aggregated Results\n{results_df.groupby(["Setting"]).mean().reset_index()}')