#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 9 11:18 2020

@author: cheonglc
"""

import math
import os
import torch
import torchvision
import argparse
import numpy as np
import pdb
import matplotlib.pyplot as plt
from DataHandler import DataHandler
from tqdm import trange
from PIL import Image
import pandas as pd
import pickle

# ---------------------------------------------------------------------------- #
#                                ARGUMENT PARSER                               #
# ---------------------------------------------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', help='Source dataset, on which we will test images from test datasets.')
parser.add_argument('--data_dir', help='Directory which contains the data for SOURCE dataset.')
parser.add_argument('--results_dir', help='Directory which contains the trained model files for the source dataset.')
'''
Required Directory Structure:
SOURCE Dataset
                Setting A
                            Seed 1
                                    model_epoch=200.pt
                            Seed 2
                                    model_epoch=200.pt
                            ...
                            Seed 5
                                    model_epoch=200.pt
                Setting B
                ...
'''
parser.add_argument('--model', help='Model type that was used in the train model files of the source dataset.')
parser.add_argument('--activation', help='Activation function that was used in the train model files of the source dataset.')
parser.add_argument('--epoch', help='Epoch of model file to use.')
parser.add_argument('--device', default='cuda', help='Device on which primary tensor arithmetic is performed.')
parser.add_argument('--graph_version', help='Version of graph to plot, switches between two types of normalizations.')
'''
Version 1 will normalize the images using the settings tuned for the SOURCE dataset that the model was trained on.
    i.e. if we put MNIST images in an Imagenette-trained model, the image data will be normalized using Imagenette settings.

Version 2 will normalize the images using the settings tuned for the TEST dataset that is being used on the model.
    i.e. if we put MNIST images in an Imagenette-trained model, the image data will be normalized using MNIST settings.
'''
parser.add_argument('--overwrite', type=int, default=0, help='Determine if the program is to overwrite existing graphs.')
param = parser.parse_args()

# ---------------------------------------------------------------------------- #
#                                    PARAMS                                    #
# ---------------------------------------------------------------------------- #

# Argument parser params.
source_dataset = param.source_dataset
data_dir = param.data_dir
results_dir = param.results_dir 
device = param.device
graph_version = int(param.graph_version) # Used to force the dataset classes to disable pre-normalization of images because we are going to do the normalization by hand
overwrite = param.overwrite
epoch = param.epoch

# These param values are hardcoded for convenience. Seed can be ignored because it only affects the random arrangement of the images when the test dataset is loaded.
param.num_class = 10
param.seed = 0
param.softmax_plotting_mode = True

# Find out which datasets will be the test datasets.
suitable_test_datasets = [d for d in [('MNIST', 32), ('CIFAR10', 32), ('Imagenette', 128)] if d[0] != source_dataset]
# Get the image size to resize to fit the source dataset, as well as the source dataset's model type.
if source_dataset == "MNIST":
    param.imsize = 32
    source_img_channels = 1
    from DatasetMnist import DatasetMnist as SourceDataset
    source_dataset_class = SourceDataset(param)
    model = source_dataset_class.getModel()
elif source_dataset == "CIFAR10":
    param.imsize = 32
    source_img_channels = 3
    from DatasetCifar import DatasetCifar as SourceDataset
    source_dataset_class = SourceDataset(param)
    model = source_dataset_class.getModel()
elif source_dataset == "Imagenette":
    param.imsize = 128
    source_img_channels = 3
    from DatasetImagenette import DatasetImagenette as SourceDataset
    source_dataset_class = SourceDataset(param)
    model = source_dataset_class.getModel()

# Matplotlib params.
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)

# Colors for nice console output.
class COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ---------------------------------------------------------------------------- #
#                             COMPILE LIST OF TESTS                            #
# ---------------------------------------------------------------------------- #

results_dir = os.path.join(results_dir, source_dataset)
settings_list = [directory_listing for directory_listing in os.listdir(results_dir) if 'NoAdv' in directory_listing]

for setting in settings_list:

    print(COLORS.OKCYAN + "Processing Setting " + setting + COLORS.END)
    
    seed_list = [s for s in os.listdir(os.path.join(results_dir, setting)) if not os.path.isfile(os.path.join(results_dir, setting, s))]
    
    for test_dataset, test_dataset_imsize in suitable_test_datasets:
        
        print(COLORS.HEADER + "Testing using test dataset " + test_dataset + COLORS.END)

        # Check if results are already there
        raw_values_path = os.path.join(results_dir, setting, f"Test with {test_dataset} 5 Seeds Version {graph_version} Raw Data.pkl")        
        if os.path.isfile(raw_values_path):
            # Results are already there
            if not overwrite:
                # If overwrite is set to 0, do not touch existing results that are there
                print(COLORS.OKGREEN + "You already have results for this, skipping to prevent overwriting..." + COLORS.END)
                continue
            else:
                # Overwrite is set to 1, so load the results that are already there and overwrite the graphs.
                print(COLORS.OKGREEN + "You already have results for this, using those..." + COLORS.END)
                with open(raw_values_path, "rb") as f:
                    softmax_values = pickle.load(f)
        else:        
            # Results are not alraedy there, we need to generate them from scratch
            if test_dataset == "MNIST":
                from DatasetMnist import DatasetMnist as TestDataset
                test_img_channels = 1
            elif test_dataset == "CIFAR10":
                from DatasetCifar import DatasetCifar as TestDataset
                test_img_channels = 3
            elif test_dataset == "Imagenette":
                from DatasetImagenette import DatasetImagenette as TestDataset
                test_img_channels = 3

            test_dataset_class = TestDataset(param)

            # Get the test split of the test dataset.
            x_test_array, y_test_array = test_dataset_class.getTest()
            

            # Resize the images to fit the source dataset.
            datahandler = DataHandler(test_dataset_class, device)
            x_test_tensor, y_test_tensor = datahandler.loadTest(x_test_array, y_test_array)
            y_test_tensor = y_test_tensor.to(device)
            num_test_images = len(y_test_tensor)

            # Transform image channels to fit source dataset.
            if source_img_channels != test_img_channels:
                
                if test_img_channels > source_img_channels:
                    # Convert from color to grayscale.
                    downscaled_x_test_tensor = torch.empty((num_test_images, 1, param.imsize, param.imsize))
                    for i in range(num_test_images):
                        # Convert to PIL to change color to grayscale
                        RGB_PIL_equivalent = torchvision.transforms.ToPILImage(mode='RGB')(x_test_tensor[i])
                        grayscale_PIL_equivalent = RGB_PIL_equivalent.convert('L')
                        # Convert back to tensor
                        downscaled_img = torchvision.transforms.ToTensor()(grayscale_PIL_equivalent)

                        # Normalize the image.
                        if graph_version is 1:
                            # Normalize the image according to the settings tuned for the SOURCE dataset
                            downscaled_img = torchvision.transforms.Normalize(source_dataset_class.mean, source_dataset_class.std)(downscaled_img)                        
                        else:
                            # Normalize the image according to the settings tuned for the TEST dataset
                            # We need to change the mean and std to grayscale versions
                            # Reference: https://pillow.readthedocs.io/en/stable/reference/Image.html
                            new_mean = [test_dataset_class.mean[0]*0.299 + test_dataset_class.mean[1]*0.587 + test_dataset_class.mean[2]*0.114]
                            new_std = [test_dataset_class.std[0]*0.299 + test_dataset_class.std[1]*0.587 + test_dataset_class.std[2]*0.114]
                            downscaled_img = torchvision.transforms.Normalize(new_mean, new_std)(downscaled_img)                        
                        
                        downscaled_x_test_tensor[i] = downscaled_img
                    
                    x_test_tensor = downscaled_x_test_tensor

                else:
                    # Duplicate grayscale channels to mimic color channels.
                    upscaled_x_test_tensor = torch.empty((num_test_images, 3, param.imsize, param.imsize))

                    for i in range(num_test_images):
                        upscaled_img = np.repeat(x_test_tensor[i], 3, axis=0)
                        
                        # Normalize the image.
                        if graph_version is 1:
                            # Normalize the image according to the settings tuned for the SOURCE dataset
                            upscaled_img = torchvision.transforms.Normalize(source_dataset_class.mean, source_dataset_class.std)(upscaled_img)                        
                        else:
                            # Normalize the image according to the settings tuned for the TEST dataset
                            upscaled_img = torchvision.transforms.Normalize(test_dataset_class.mean, test_dataset_class.std)(upscaled_img)  

                        upscaled_x_test_tensor[i] = upscaled_img
                    
                    x_test_tensor = upscaled_x_test_tensor

            # Only now do we move x_test_tensor to GPU because numpy needs them on CPU.
            x_test_tensor = x_test_tensor.to(device)

            # Create the array that will hold the softmax values for averaging.
            softmax_values = []

            for seed_index, seed_string in enumerate(seed_list):

                seed = int(seed_string[-5:])

                print(COLORS.OKBLUE + "Processing " + seed_string + COLORS.END)

                # Find the path to the model file for the current setting and seed.
                model_dir = os.path.join(results_dir, setting, seed_string)
                model_file = [filename for filename in os.listdir(model_dir) if f'model_epoch={epoch}.pt' in filename][0]
                model_path = os.path.join(model_dir, model_file)

                # Load the model.
                model.load_state_dict(torch.load(model_path)["model_state_dict"])
                
                # Switch to inference mode.
                model.eval()

                # Pass the test images through the model.
                for i in trange(num_test_images):
                    # Get the current image and reshape it to include the dimension for batch size.
                    curr_img = x_test_tensor[i].reshape(1, source_img_channels, param.imsize, param.imsize)
                    # Perform the forward pass and pass them through softmax. Intentionally specify no gradient computation because we are only interested in inference
                    with torch.no_grad():
                        preds = model(curr_img).softmax(axis=-1)
                    # Append the softmax values to the array for graph plotting purposes.
                    softmax_values.append(preds.squeeze().tolist())
            
            # Transpose so we get all the softmax values in 10 classes
            softmax_values = np.transpose(np.array(softmax_values))

        print(COLORS.OKGREEN + "Proceeding to create graph." + COLORS.END)
        
        fig, ax = plt.subplots()

        # Create the chart
        x_values = range(param.num_class)
        bar_heights = [np.mean(class_scores) for class_scores in softmax_values]
        errors = [np.std(class_scores) for class_scores in softmax_values]
        ax.bar(x_values, bar_heights, yerr=errors, capsize=10)
        
        # Create the labels
        ax.set_xlabel("Class")
        ax.set_xticks(range(param.num_class))
        ax.set_ylabel("Average Softmax Logits")
        ax.set_title("Graph of Average Softmax Logits against Class")

        # Ensure graphs fit properly in the space
        ax.set_ylim([-0.3,1.3])
        plt.tight_layout()

        # Save the graph
        graph_path = os.path.join(results_dir, setting, f"Test with {test_dataset} 5 Seeds Version {graph_version}.png")
        plt.savefig(graph_path)

        print(COLORS.OKGREEN + "Graph saved to " + graph_path + COLORS.END)
        
        # Log results
        results_df = pd.DataFrame([], columns=["Class", "Mean Softmax Logits", "Std"])
        for c in range(param.num_class):
            results_df = results_df.append({"Class": c, "Mean Softmax Logits": np.mean(softmax_values[c]), "Std": np.std(softmax_values[c])}, ignore_index=True)
        print(results_df)
        log_path = os.path.join(results_dir, setting, f"Test with {test_dataset} 5 Seeds Version {graph_version} Log.txt")
        with open(log_path, "w") as f:
            f.write(results_df.to_string())
            f.write("\n")
            f.write("========================================")
            f.write("\n")
            f.write("Overall Logits Mean: " + str(results_df["Mean Softmax Logits"].mean()))
            f.write("\n")
            f.write("Overall Logits Std: " + str(np.std(results_df["Mean Softmax Logits"])))
            f.close()
        print(COLORS.OKGREEN + "Results logged to " + log_path + COLORS.END)

        # Save raw softmax values
        with open(raw_values_path, "wb") as f:
            pickle.dump(softmax_values, f)
        
        print(COLORS.OKGREEN + "Raw data saved to " + raw_values_path + COLORS.END)
