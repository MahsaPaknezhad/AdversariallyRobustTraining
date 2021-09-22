#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:14:15 2020

@author: amadeusaw
"""

import argparse
import numpy as np
from tqdm import tqdm

from FoolboxAttack import FoolboxAttack, AttackLogging
from DataHandler import DataHandler
from Seed import getSeed

import matplotlib.pyplot as plt
import os
from scipy import stats

'''
The following directory structure must be present for code to run
Legends: {X} means that X is a parameter to be passed in before running this code
Usage instruction: To get this directory structure:
    After completing training code, rename the "time" folder to the setting used 
    For example, suppose 'time' folder is '05_06_2020-15_55_10'
    Just rename this folder to  'Lambda=0', if this is the setting used for the experiment there
    To check the setting, open the Results.json file in that 'time' folder

Required Directory Structure:
{method}__
        | {dataset}__
                    | Seed A____
                                | Setting 1_____
                                                | model_epoch=10.pt
                                                | model_epoch=20.pt
                                                | ...
                                                | model_epoch={epoch}.pt
                                | Setting 2_____
                                                | model_epoch=10.pt
                                                | model_epoch=20.pt
                                                | ...
                                                | model_epoch={epoch}.pt
                                | ...
                                | Setting N_____
                                                | model_epoch=10.pt
                                                | model_epoch=20.pt
                                                | ...
                                                | model_epoch={epoch}.pt
                 ___
                    | Seed B____
                                | Setting 1_____
                                                | model_epoch=10.pt
                                                | model_epoch=20.pt
                                                | ...
                                                | model_epoch={epoch}.pt
                                | Setting 2_____
                                                | model_epoch=10.pt
                                                | model_epoch=20.pt
                                                | ...
                                                | model_epoch={epoch}.pt
                                | ...
                                | Setting N_____
                                                | model_epoch=10.pt
                                                | model_epoch=20.pt
                                                | ...
                                                | model_epoch={epoch}.pt
                 ...
                 ...
                 ___
                    | Seed Z____
                                | Setting 1_____
                                                | model_epoch=10.pt
                                                | model_epoch=20.pt
                                                | ...
                                                | model_epoch={epoch}.pt
                                | Setting 2_____
                                                | model_epoch=10.pt
                                                | model_epoch=20.pt
                                                | ...
                                                | model_epoch={epoch}.pt
                                | ...
                                | Setting 3_____
                                                | model_epoch=10.pt
                                                | model_epoch=20.pt
                                                | ...
                                                | model_epoch={epoch}.pt
'''

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (5, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)
print(os.getcwd())

parser = argparse.ArgumentParser()
############################## PARSER ########################################parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default= 'Imagenette')
parser.add_argument('--data_dir', default='../data', help = 'Folder where the dataset is located')
parser.add_argument('--method', default='../output/Adversarial/Imagenette/Reg_vs_Adv&Reg/', help = "The premise to be used, for documentation purposes")
parser.add_argument('--model', default='xresnet18')
parser.add_argument('--epoch', default= 200, help = 'Weight of the model at that epoch to test')
parser.add_argument('--activation', default='mish')
parser.add_argument('--batch_size', type=int, default=16, help = 'batch size to do attack')
parser.add_argument('--attack_method', default='fgsm', help = 'Either fgsm, pgd, deepfooll2 or deepfoollinf')
parser.add_argument('--min_epsilon', type = int, default=0, help = 'An integer. This should be before division by 255.')
parser.add_argument('--max_epsilon', type = int, default=20, help = 'An integer. This should be before division by 255.')
parser.add_argument('--step_epsilon', type = int, default=1, help = 'An integer. This should be before division by 255.')
parser.add_argument('--imsize', type=int, default=128, help = 'Only necessary if dataset is Imagenette')
parser.add_argument('--seed', type=int, default=0, help = 'Seed to run experiment. Ignored if time != -1')
parser.add_argument('--time', type=int, default=-1,   help = "`Seed` to generate actual seed to run experiment. Ignored if -1")
parser.add_argument('--device', default = 'cuda')
param = parser.parse_args()
################################# PARSER #####################################
'''parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default= 'CIFAR10')
parser.add_argument('--data_dir', default='../data', help = 'Folder where the dataset is located')
parser.add_argument('--method', default='Adversarial/CIFAR10/Epsilon_2&3&10_Files', help = "The premise to be used, for documentation purposes")
parser.add_argument('--model', default='resnet9')
parser.add_argument('--epoch', default= 200, help = 'Weight of the model at that epoch to test')
parser.add_argument('--activation', default='celu')
parser.add_argument('--batch_size', type=int, default=32, help = 'batch size to do attack')
parser.add_argument('--attack_method', default='fgsm', help = 'Either fgsm, pgd, deepfooll2 or deepfoollinf')
parser.add_argument('--min_epsilon', type = int, default=0, help = 'An integer. This should be before division by 255.')
parser.add_argument('--max_epsilon', type = int, default=16, help = 'An integer. This should be before division by 255.')
parser.add_argument('--step_epsilon', type = int, default=1, help = 'An integer. This should be before division by 255.')
parser.add_argument('--imsize', type=int, default=128, help = 'Only necessary if dataset is Imagenette')
parser.add_argument('--seed', type=int, default=0, help = 'Seed to run experiment. Ignored if time != -1')
parser.add_argument('--time', type=int, default=-1, help = "`Seed` to generate actual seed to run experiment. Ignored if -1")
parser.add_argument('--device', default = 'cuda')
param = parser.parse_args()'''
############################### PARAMETERS ###################################
'''parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default= 'MNIST')
parser.add_argument('--data_dir', default='../data', help = 'Folder where the dataset is located')
parser.add_argument('--method', default='NoDefence', help = "The premise to be used, for documentation purposes")
parser.add_argument('--model', default='basicmodel')
parser.add_argument('--epoch', default= 100, help = 'Weight of the model at that epoch to test')
parser.add_argument('--activation', default='sigmoid')
parser.add_argument('--batch_size', type=int, default=32, help = 'batch size to do attack')
parser.add_argument('--attack_method', default='deepfoollinf', help = 'Either fgsm, pgd, deepfooll2 or deepfoollinf')
parser.add_argument('--min_epsilon', type = int, default=0, help = 'An integer. This should be before division by 255.')
parser.add_argument('--max_epsilon', type = int, default=20, help = 'An integer. This should be before division by 255.')
parser.add_argument('--step_epsilon', type = int, default=1, help = 'An integer. This should be before division by 255.')
parser.add_argument('--imsize', type=int, default=128, help = 'Only necessary if dataset is Imagenette')
parser.add_argument('--seed', type=int, default=0, help = 'Seed to run experiment. Ignored if time != -1')
parser.add_argument('--time', type=int, default=-1, help = "`Seed` to generate actual seed to run experiment. Ignored if -1")
parser.add_argument('--device', default = 'cuda')
param = parser.parse_args()'''
################################# PARSER #####################################
'''parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default= 'SVHN')
parser.add_argument('--data_dir', default='../data/SVHN', help = 'Folder where the dataset is located')
parser.add_argument('--method', default='SVHN/Premise2/', help = "The premise to be used, for documentation purposes")
parser.add_argument('--model', default='svhn_mod')
parser.add_argument('--epoch', default= 200, help = 'Weight of the model at that epoch to test')
parser.add_argument('--activation', default='relu')
parser.add_argument('--batch_size', type=int, default=32, help = 'batch size to do attack')
parser.add_argument('--attack_method', default='fgsm', help = 'Either fgsm or pgd')
parser.add_argument('--min_epsilon', type = int, default=0, help = 'An integer. This should be before division by 255.')
parser.add_argument('--max_epsilon', type = int, default=16, help = 'An integer. This should be before division by 255.')
parser.add_argument('--step_epsilon', type = int, default=1, help = 'An integer. This should be before division by 255.')
parser.add_argument('--imsize', type=int, default=128, help = 'Only necessary if dataset is Imagenette')
parser.add_argument('--seed', type=int, default=0, help = 'Seed to run experiment. Ignored if time != -1')
parser.add_argument('--time', type=int, default=-1, help = "`Seed` to generate actual seed to run experiment. Ignored if -1")
parser.add_argument('--device', default = 'cuda')
param = parser.parse_args()'''

if param.time != -1:
    param.seed = getSeed(time = param.time)
device = param.device
batch_size = param.batch_size

dataset = param.dataset

epoch = param.epoch
attack_method = param.attack_method
min_epsilon = param.min_epsilon
max_epsilon = param.max_epsilon
step_epsilon = param.step_epsilon

method = param.method
target_path = os.path.join('../output',method)
############################# PARAMETERS #####################################


############################# Initiate Class #################################
if dataset == "MNIST":
    from DatasetMnist import DatasetMnist as Dataset
elif dataset == "CIFAR10":
    from DatasetCifar import DatasetCifar as Dataset
elif dataset == "Imagenette":
    from DatasetImagenette import DatasetImagenette as Dataset
elif dataset == "SVHN":
    from DatasetSvhn2 import DatasetSvhn as Dataset

dataset_class = Dataset(param)

x_test_array, y_test_array = dataset_class.getTest()
datahandler= DataHandler(dataset_class, device)
x_test_tensor, y_test_tensor = datahandler.loadTest(x_test_array, y_test_array)

x_test_tensor = x_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

filename_list = os.listdir(target_path)
seed_list = []
for filename in filename_list:
    if os.path.isfile(os.path.join(target_path, filename)):
        continue
    else:
        seed_list.append(filename)

#seed_list = [seed_list[0]]

setting_list = os.listdir(os.path.join(target_path, seed_list[0]))
#for i in range(len(setting_list)): setting_list[i] = int(setting_list[i])
#setting_list.sort()
model_class = dataset_class.getModel()

epsilons = np.arange(min_epsilon, max_epsilon, step_epsilon) / 255.

numSeeds = len(seed_list)
numEpsilons = len(epsilons)
numModels = len(setting_list)

clean_acc_array = np.zeros((numSeeds, numModels))
robust_acc_array = np.zeros((numSeeds, numModels, numEpsilons))
fa = FoolboxAttack(batch_size, device, dataset_class.mean, dataset_class.std, attack_method)
proportions = []

for i, seed in enumerate(tqdm(seed_list)):
    logs = AttackLogging(numEpsilons)
    current_path = os.path.join(target_path, seed)

    for setting_name in setting_list:
        currPath = os.path.join(current_path, str(setting_name))
        model_name = 'model_epoch={}.pt'.format(epoch)
        fmodel = fa.FoolboxModel(model_class, model_name, currPath)
        clean_acc, correct_idx = fa.getClean(fmodel, x_test_tensor, y_test_tensor)
        print(currPath + ' ' + str(clean_acc))
        logs.logClean(str(setting_name), clean_acc, correct_idx)


    x_test_selected = x_test_tensor[logs.correct_index]
    y_test_selected = y_test_tensor[logs.correct_index]
    proportions.append(len(logs.correct_index) / len(x_test_array) * 100)

    assert len(x_test_selected) == len(y_test_selected)

    for setting_name in setting_list:
        currPath = os.path.join(current_path, str(setting_name))
        model_name = 'model_epoch={}.pt'.format(epoch)
        fmodel = fa.FoolboxModel(model_class, model_name, currPath)
        robust_acc = fa.getRobust(fmodel, x_test_selected, y_test_selected, epsilons)
        logs.logRobust(setting_name, robust_acc)


    logs.saveLogs(seed, target_path)
    clean_acc_array[i] = np.array([logs.clean[str(setting)] for setting in setting_list])
    robust_acc_array[i] = logs.robust[1:]


proportions = np.array(proportions).flatten()
mean_proportions = np.mean(proportions, axis=0)
ste_proportions = stats.sem(proportions, axis=0)

mean_clean_acc = np.mean(clean_acc_array, axis = 0)
ste_clean_acc = stats.sem(clean_acc_array, axis = 0)

mean_robust_acc = np.mean(robust_acc_array, axis = 0)
ste_robust_acc =  stats.sem(robust_acc_array, axis = 0)



colors = ['seagreen', '#3498DB', '#FF5733', '#8E44AD', '#FFC300', '#C70039', '#DAF7A6', '#34495E']
style = ['-^', '-^', '-^', '-^', '-^', '-^']
#colors = ['seagreen', 'seagreen', '#3498DB',  '#3498DB',  '#FF5733', '#FF5733', '#8E44AD', '#8E44AD']
#style = ['-^', '--^', '-^', '--^', '-^', '--^']
legend= '$N_u = %d$'

fig, ax = plt.subplots(figsize = (10, 6))
for i, setting in enumerate(setting_list):
    plt.plot(epsilons, mean_robust_acc[i], style[i], color = colors[i], label=setting)
    plt.fill_between(epsilons, mean_robust_acc[i] - ste_robust_acc[i], mean_robust_acc[i] + ste_robust_acc[i],
                            alpha = 0.5, edgecolor = colors[i], facecolor = colors[i])

ax.set_xticks(epsilons)
plt.grid(True)
plt.xticks(rotation=45)
plt.ylim([0, 1])

np.savez(os.path.join(target_path,'{}_{}.npz'.format(dataset, attack_method)),
              setting_list = setting_list,
              epsilons=epsilons,mean_robust_acc=mean_robust_acc,
              ste_robust_acc=ste_robust_acc, mean_proportions=mean_proportions,
              ste_proportions=ste_proportions)


#labels = ["{}/255".format(int(255*x)) for x in epsilons]
labels = ["%.03f" % x for x in epsilons]

ax.set_xticklabels(labels)

plt.legend(loc='upper right')
plt.xlabel('epsilons')
plt.ylabel('Robust accuracy')
plt.tight_layout()
plt.title(r"{0:.2f}% ".format(mean_proportions) + u"\u00B1" + r" {0:.2f}% Correctly Labeled".format(ste_proportions))
plt.show()
plt.savefig(os.path.join(target_path, '{}-5runs.png'.format(attack_method.upper())), dpi=600, bbox_inches='tight')
