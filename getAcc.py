
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import math
import ast
import pdb

from tqdm import tqdm, trange
from foolbox import attacks, PyTorchModel
from DataHandler import DataHandler
from Seed import getSeed
from DeepFool import deepfool
from scipy.stats import sem
from Logging import info, success, warn

# ---------------------------------------------------------------------------- #
#                                ARGUMENT PARSER                               #
# ---------------------------------------------------------------------------- #

parser = argparse.ArgumentParser()
# Dataset parameters.
parser.add_argument('--dataset', type=str, choices=['MNIST', 'CIFAR10', 'Imagenette'], help='Dataset to be used for plotting of robust accuracy')
parser.add_argument('--data_dir', type=str, default='../data', help='Directory to get dataset from')
parser.add_argument('--output_dir', type=str, default='../output/', help='Directory to output results')
parser.add_argument('--imsize', type=int, help='Image size, set to 32 for MNIST and CIFAR10, set to 128 for Imagenette')
parser.add_argument('--seed', type=int, default=0, help='Seed to be used during experiment')
parser.add_argument('--time', type=int, default=-1, help='Seed to set the seed to be used during training, not used by default')
# Experiment setting parameters.
parser.add_argument('--settings_list', type=str, help='Comma-separated list of settings to consider.')
parser.add_argument('--seed_list', default='27432,30416,48563,51985,84216', type=str, help='Comma-separated list of seeds to consider.')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to perform experiment on')
# Model parameters.
parser.add_argument('--model', help='Model type, set to basicmodel for MNIST, resnet9 for CIFAR10 and xresnet18 for Imagenette. Check LoadModel.py for the list of supported models.')
parser.add_argument('--activation', help='Activation function for the model, set to sigmoid for MNIST, celu for CIFAR10 and mish for Imagenette')
parser.add_argument('--epochs', type=str, help='Comma-separated list of epoch numbers of the model files to test, in the order of each setting respectively.')
# Attack parameters.
parser.add_argument('--attack_method', type=str, choices=['FGSM', 'PGD'], help='Select which attack to use.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to perform attack.')
parser.add_argument('--min_epsilon', type=float, help='Starting value for perturbation magnitude. This should be before division by 255. Set to 12.75 for MNIST, and 0.0 for CIFAR10 and Imagenette.')
parser.add_argument('--max_epsilon', type=float, help='Ending value for perturbation magnitude. This should be before division by 255. Set to 191.25 for MNIST, 16.0 for CIFAR10, and 20 for Imagenette.')
parser.add_argument('--step_epsilon', type=float, help='Step value for perturbation magnitude. This should be before divison by 255. Set to 12.75 for MNIST, and 1.0 for CIAR10 and Imagenette.')
# Graph plotting parameters.
parser.add_argument('--colors', type=str, help='Comma-separated list of color codes WITHOUT THE HEX SYMBOL to plot the curves. This runs in the same order as settings_list, seed_list and epochs.')
parser.add_argument('--styles', type=str, help='Comma-separated list of matplotlib styles to plot the curves. This runs in the same order as the settings_list, seed_list and epochs.')
parser.add_argument('--legend_labels', type=str, default='', help='Comma-seaprated list of labels to give the legend. This runs in the same order as your settings. Leaving this blank will assume that you want to use the setting name as the legend label.')
param = parser.parse_args()

# ---------------------------------------------------------------------------- #
#                                    PARAMS                                    #
# ---------------------------------------------------------------------------- #

# Dataset parameters.
dataset = param.dataset
data_dir = param.data_dir
output_dir = param.output_dir
if param.time != -1:
    param.seed = getSeed(time = param.time)
# Experiment setting parameters.
settings_list = param.settings_list.split(',')
seed_list = [f'Seed_{s}' for s in param.seed_list.split(',')]
device = param.device
# Model parameters.
epochs = param.epochs.split(',')
# Attack parameters.
attack_method = param.attack_method
batch_size = param.batch_size
min_epsilon = param.min_epsilon
max_epsilon = param.max_epsilon
step_epsilon = param.step_epsilon
# Graph plotting parameters.
colors = param.colors
styles = param.styles
legend_labels = settings_list if param.legend_labels == '' else param.legend_labels

target_path = os.path.join(output_dir, dataset)
results_dir = os.path.join(target_path, 'Robust_Accuracy_Results', attack_method)

# Matplotlib parameters.
rc_params = {'legend.fontsize': 'x-large',
          'figure.figsize': (5, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(rc_params)

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
datahandler= DataHandler(dataset_class, device)
x_test_tensor, y_test_tensor = datahandler.loadTest(x_test_array, y_test_array)
test_size = len(x_test_tensor)

x_test_tensor = x_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# ---------------------------------------------------------------------------- #
#                              LOAD DATA IF EXISTS                             #
# ---------------------------------------------------------------------------- #

if os.path.exists(results_dir) and os.path.isfile(os.path.join(results_dir, 'proportions_and_epsilons.pkl')):
    info(f'It seems you already have results logged in {results_dir}, loading them...')
    with open(os.path.join(results_dir, 'proportions_and_epsilons.pkl'), 'rb') as f:
        data = pickle.load(f)
        proportions = data['proportions']
        epsilons = data['epsilons']
    clean_stage_df = pd.read_csv(os.path.join(results_dir, 'clean_stage.csv'))
    robust_stage_df = pd.read_csv(os.path.join(results_dir, 'robust_stage.csv'))
    success(f'Successfully loaded results from {results_dir}')
else:
    warn(f'No prior results found in {results_dir}, starting from scratch...')
    # Create the data structures that will hold the results for graph plotting later
    clean_stage_df = pd.DataFrame(columns=['Seed', 'Correct Indices'])
    robust_stage_df = pd.DataFrame(columns=['Setting', 'Seed', 'Robust Accuracy'])
    proportions = []

# Instantiate model class
model = dataset_class.getModel()

# Instantiate attack method
if attack_method == 'FGSM':
    attack = attacks.FGSM()
elif attack_method == 'PGD':
    attack = attacks.PGD()

# Instantiate epsilons array
epsilons = np.arange(min_epsilon, max_epsilon, step_epsilon) / 255.

# ---------------------------------------------------------------------------- #
#                               START EXPERIMENT                               #
# ---------------------------------------------------------------------------- #

for j, seed_string in enumerate(seed_list):

    info(f'Processing {seed_string}')

    epoch = epochs[j]
    # Skip the clean accuracy computation stage if the results are already available
    if clean_stage_df.empty or not (clean_stage_df['Seed'] == seed_string).any():
        # Perform the clean accuracy computation stage
        info('Clean Accuracy Stage')

        # This set will contain the indices of the images that all settings trained with this seed classified correctly.
        correct_indices = set()

        for setting in settings_list:

            info(f'Processing {setting}')

            # Load the model
            model_path = os.path.join(target_path, setting, seed_string, f'model_epoch={epoch}.pt')
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            model.eval().to(device)
            success(f'Successfully loaded model file from {model_path}')

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
            candidate_set = set(correct_index.tolist())
            if len(correct_indices) == 0:
                correct_indices = candidate_set
            else:
                correct_indices = correct_indices.intersection(candidate_set)

        clean_stage_df = clean_stage_df.append({
            'Seed': seed_string,
            'Correct Indices': list(correct_indices)
        }, ignore_index=True)

    else:
        warn(f'You already have clean accuracy results for {seed_string}, skipping...')

    # Perform the robust accuracy computation stage
    info('Robust Accuracy Stage')

    for setting in settings_list:

        info(f'Processing {setting}')
        if robust_stage_df.empty or not ((robust_stage_df['Setting'] == setting) & (robust_stage_df['Seed'] == seed_string)).any():
            # Select only the images which were classified correctly by the model
            correct_index = clean_stage_df[clean_stage_df['Seed'] == seed_string]['Correct Indices'].values[0]
            if type(correct_index) is not type([]):
                correct_index = ast.literal_eval(correct_index)
            x_test_selected = x_test_tensor[correct_index]
            y_test_selected = y_test_tensor[correct_index]
            proportions.append(len(correct_index) / len(x_test_array) * 100)

            # Do a check that the number of selected images should be the same as the number of selected labels
            assert len(x_test_selected) == len(y_test_selected)

            # Load the model
            model_path = os.path.join(target_path, setting, seed_string, f'model_epoch={epoch}.pt')
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            model.eval().to(device)
            fmodel = PyTorchModel(model, bounds=(0, 1), device=device, preprocessing=dict(mean=dataset_class.mean, std=dataset_class.std, axis=-3))
            success(f'Successfully loaded model file from {model_path}')

            success_count = np.zeros(len(epsilons))

            # Perform the attack
            for batch_idx in trange(math.ceil(len(x_test_selected)/batch_size)):
                start_batch = batch_idx * batch_size
                end_batch = min((batch_idx+1)*batch_size, test_size)

                # Get batch of image data and label data.
                x_batch = torch.squeeze(x_test_selected[start_batch:end_batch], 0).to(device)
                y_batch = torch.squeeze(y_test_selected[start_batch:end_batch], 0).to(device)

                # Generate perturbed image.
                _, _, is_success = attack(fmodel, x_batch, y_batch, epsilons=epsilons)
                is_success = is_success.cpu().numpy()

                assert is_success.shape == (len(epsilons), len(x_batch))
                assert is_success.dtype == np.bool

                success_count += np.sum(is_success, axis=-1)

            # Compute robust accuracy.
            robust_accuracy = 1 - ( (success_count * 1.0) / test_size)
            success(str(robust_accuracy))

            # Log the values into pandas dataframe
            robust_stage_df = robust_stage_df.append({
                'Setting': setting,
                'Seed': seed_string,
                'Robust Accuracy': robust_accuracy.tolist()
            }, ignore_index=True)

            # ---------------------------------------------------------------------------- #
            #                              LOG & SAVE PROGRESS                             #
            # ---------------------------------------------------------------------------- #

            info('Proceeding to save results.')

            # Create results directory if it does not exist
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # Pickle a dictionary of the proportions and epsilons
            with open(os.path.join(results_dir, f'proportions_and_epsilons.pkl'), 'wb') as f:
                pickle.dump({
                    'proportions': proportions,
                    'epsilons': epsilons
                }, f)

            clean_stage_df.to_csv(os.path.join(results_dir, 'clean_stage.csv'), index=False)
            robust_stage_df.to_csv(os.path.join(results_dir, 'robust_stage.csv'), index=False)

            success(f'Results saved to {results_dir}.')
        
        else:
            warn(f'You already have robust accuracy results for {setting}/{seed_string}, skipping...')

# ---------------------------------------------------------------------------- #
#                                  PLOT GRAPH                                  #
# ---------------------------------------------------------------------------- #

info('Plotting graph.')

# Compute proportions
proportions = np.array(proportions).flatten()
mean_proportions = np.mean(proportions, axis=0)
ste_proportions = sem(proportions, axis=0)

# Create the actual graph. Every curve represents the aggregated results of 5 seeds for 1 particular setting.
fig, ax = plt.subplots()
for i, setting in enumerate(settings_list):
    robust_accuracies_array = np.zeros((len(seed_list), len(epsilons)))
    for j, seed in enumerate(seed_list):
        robust_accuracies_array[j] = ast.literal_eval(robust_stage_df[(robust_stage_df['Setting'] == setting) & (robust_stage_df['Seed'] == seed)]['Robust Accuracy'].values[0])
    averaged_accuracies = robust_accuracies_array.mean(axis=0)
    ste = sem(averaged_accuracies)
    plt.plot(epsilons, averaged_accuracies, styles[i], color=colors[i], label=legend_labels[i])
    plt.fill_between(epsilons, averaged_accuracies - ste, averaged_accuracies + ste, alpha=0.5, edgecolor=colors[i], facecolor=colors[i])

ax.set_xticks(epsilons)
plt.grid(True)
plt.xticks(rotation=45)
plt.ylim([0, 1])

labels = [f'{int(x*255)}' for x in epsilons]

ax.set_xticklabels(labels)

plt.legend(loc='lower left')
plt.xlabel('Epsilons')
plt.ylabel('Robust Accuracy')
plt.tight_layout()
plt.title(r'{0:.2f}% '.format(mean_proportions) + u'\u00B1' + r' {0:.2f}% Correctly Labeled'.format(ste_proportions))
plt.show()

graph_path = os.path.join(results_dir, 'graph.png')
plt.savefig(graph_path, dpi=600, bbox_inches='tight')

success(f'Graph saved to {graph_path}.')
