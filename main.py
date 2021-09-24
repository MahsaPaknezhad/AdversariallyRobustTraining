#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Main file used to train models.

import math
import os
import torch
import argparse
import json
import numpy as np
import re
import torch.nn as nn

from tqdm import trange
from Optimizers import Optimizers
from NoiseGenerator import NeighborGenerator, UnlabeledGenerator
from LossFunction import CELoss, GRLoss
from DataHandler import DataHandler
from Seed import getSeed
from Logging import Logging, info, success, warn
from FGSM import fgsm
from jacobian import JacobianReg
from getScheduler import lr_scheduler

# ---------------------------------------------------------------------------- #
#                                ARGUMENT PARSER                               #
# ---------------------------------------------------------------------------- #

parser = argparse.ArgumentParser()
# Resume parameters.
parser.add_argument('--resume', type=int, default=0, help='Whether to resume training or not')
parser.add_argument('--resume_epoch', type=int, default=-1, help='Epoch to resume training from, set to -1 to resume from latest epoch')
# Dataset parameters.
parser.add_argument('--dataset', type=str, default = 'Imagenette', choices=['MNIST', 'CIFAR10', 'Imagenette'], help='Dataset to be used for training')
parser.add_argument('--setting', default='imagenette_jacobian',type=str, help='Name of your setting directory, leave blank for an automatically generated setting name that includes experiment parameters.')
parser.add_argument('--data_dir', type=str, default='/media/mahsa/Element/Regularization/data', help='Directory to download dataset to')
parser.add_argument('--output_dir', type=str, default='/media/mahsa/Element/Regularization/output/', help='Directory to output results')
parser.add_argument('--num_class', type=int, default=10, help = 'Number of classes in the dataset')
parser.add_argument('--num_train', type=int, default=-1, help='Number of train images per class, set to -1 to use full dataset')
parser.add_argument('--num_val', type=int, default=-1, help='Number of validation images per class, set to -1 to use full dataset')
parser.add_argument('--imsize', type=int, default=128, help='Image size, set to 32 for MNIST and CIFAR10, set to 128 for Imagenette')
parser.add_argument('--seed', type=int, default=27432, help='Seed to be used during training, choose from 27432, 30416, 48563, 51985 or 84216 to reproduce results in the paper')
parser.add_argument('--time', type=int, default=-1, help='Seed to set the seed to be used during training, not used by default')
# Jacobian parameters.
parser.add_argument('--jacobian', type=int, default=0, help='Whether to use jacobian regularization or not.')
# Gradient regularization parameters.
parser.add_argument('--grad_reg_lambda', type=float, default=0, help='Lambda for gradient regularization, set to 100,000 for MNIST, 10,000 for CIFAR10 and 10,000 Imagenette, set to 1,000,000 for Imagenette when doing both adversarial training and gradient regularization')
parser.add_argument('--num_unlabeled_per_labeled', type=int, default=0, help='Number of unlabeled points to be made per labeled point')
parser.add_argument('--unlabeled_noise_std', type=float, default=0.138687, help='Standard deviation to make unlabeled points, set to 0.0126 for MNIST, 0.23769 for CIFAR10 and 0.138687 for Imagenette')
parser.add_argument('--num_neighbor_per_anchor', type=int, default=0, help='Number of neighbor points to be make per anchor point')
parser.add_argument('--neighbor_noise_std', type=float, default=0.0138687, help='Standard deviation to make neighbors, set to 0.0126 for MNIST, 0.023769 for CIFRAR10 and 0.0138687 for Imagenette')
parser.add_argument('--inject_noise', type=int, default=0, help='Inject noise in the middle of the network')
# Adversarial training parameters.
parser.add_argument('--adversarial', type=int, default=0, help='Whether to perform adversarial training or not')
parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon to use during the FGSM attack, only used if adversarial=1')
parser.add_argument('--adv_ratio', type=float, default=0.3, help='The weight that is used in the calculation of the weighted average of clean image loss and perturbed image loss')
# Training parameters.
parser.add_argument('--model', type=str, default='XResNet18', help='Model type, set to BasicModel for MNIST, ResNet9 for CIFAR10 and XResNet18 for Imagenette. Check LoadModel.py for the list of supported models.')
parser.add_argument('--activation', type=str, default='mish', help='Activation function for the model, set to sigmoid for MNIST, celu for CIFAR10 and mish for Imagenette')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate is 0.001 for MNIST dataset, 0.0001 for CIFAR10 and Imagenette datasets')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay is 0 for MNIST dataset and 0.0001 for CIFAR10 and Imagenette datasets')
parser.add_argument('--linear_epoch', type=int, default=100, help='Number of epochs for which the learning rate remains constant, applicable only to the learning rate scheduler')
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train the network')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size to do training')
parser.add_argument('--log_freq', type=int, default=10, help='Frequency of logging and saving model files')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to perform training on')
param = parser.parse_args()

# ---------------------------------------------------------------------------- #
#                                  PARAMETERS                                  #
# ---------------------------------------------------------------------------- #

# Resume parameters.
resume = param.resume
resume_epoch = param.resume_epoch
setting = (f'num_train={param.num_train}'
        f'&lambda={param.grad_reg_lambda}'
        f'&unlabeled={param.num_unlabeled_per_labeled}'
        f'&neighbor={param.num_neighbor_per_anchor}'
        f'&adversarial={param.adversarial}'
        f'&epoch={param.num_epochs}')
resume_dir = os.path.join(param.output_dir, param.dataset, param.setting, f'Seed_{param.seed}')
json_log_path = os.path.join(resume_dir, 'result.json')
if resume:
    new_num_epochs = param.num_epochs
    if resume_epoch != -1:
        info(f'Resuming from epoch {resume_epoch}.')
        if os.path.exists(json_log_path):
            with open(json_log_path) as f:
                result_dict = json.load(f)
                result_dict['resume_epoch'] = resume_epoch
                success('Successfully read result JSON file.')
        else:
            warn('resume_dir not found. Starting from epoch 0.')
    else:
        # Find the latest epoch
        if os.path.exists(json_log_path):
            model_epoch_list = [fname for fname in os.listdir(resume_dir) if '.pt' in fname]
            highest_epoch_num = 0
            for fname in model_epoch_list:
                curr_epoch_num = int(re.search('=(\d+)\.', fname).groups()[0])
                if curr_epoch_num > highest_epoch_num:
                    highest_epoch_num = curr_epoch_num
            resume_epoch = highest_epoch_num
            info(f'Resuming from epoch {resume_epoch}.')
            with open(json_log_path) as f:
                result_dict = json.load(f)
                result_dict['result_dir'] = resume_dir
                result_dict['resume_epoch'] = resume_epoch
                result_dict['log_freq'] = param.log_freq
                result_dict['ce_loss'] = result_dict['ce_loss'][:resume_epoch]
                result_dict['gr_loss'] = result_dict['gr_loss'][:resume_epoch]
                result_dict['train_acc'] = result_dict['train_acc'][:resume_epoch]
                result_dict['val_loss'] = result_dict['val_loss'][:resume_epoch]
                result_dict['val_acc'] = result_dict['val_acc'][:resume_epoch]
                success('Successfully read result JSON file.')
        else:
            warn('resume_dir not found. Starting from epoch 0.')

# Dataset parameters.
dataset = param.dataset
setting = param.setting
output_dir = param.output_dir
num_class = param.num_class
num_train = param.num_train
seed = param.seed
if param.time != -1:
    param.seed = getSeed(time=param.time)
#Jacobian parameters.
jacobian = param.jacobian
# Gradient regularization parameters.
grad_reg_lambda = param.grad_reg_lambda
num_unlabeled_per_labeled = param.num_unlabeled_per_labeled
unlabeled_noise_std = param.unlabeled_noise_std
num_neighbor_per_anchor = param.num_neighbor_per_anchor
neighbor_noise_std = param.neighbor_noise_std
# Adversarial training parameters.
adversarial = param.adversarial
epsilon = param.epsilon
adv_ratio = param.adv_ratio
# Training paramteters.
weight_decay = param.weight_decay
linear_epoch = param.linear_epoch
num_epochs = param.num_epochs
batch_size = param.batch_size
log_freq = param.log_freq
device = param.device
inject_noise = param.inject_noise

if inject_noise and adversarial:
    raise Exception('Sorry, intermediate noise injection + adversarial training is currently unsupported.')
    exit
if inject_noise and param.model not in ['BasicModel', 'ResNet9', 'XResNet18']:
    raise Exception('Noise injection can only be used with BasicModel, ResNet9, and XResNet18.')
    exit

# ---------------------------------------------------------------------------- #
#                       MAKING THE PROGRAM DETERMINISTIC                       #
# ---------------------------------------------------------------------------- #

np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------------------- #
#                              INSTANTIATE CLASSES                             #
# ---------------------------------------------------------------------------- #

if dataset == 'MNIST':
    from DatasetMNIST import DatasetMNIST as Dataset
elif dataset == 'CIFAR10':
    from DatasetCIFAR10 import DatasetCIFAR10 as Dataset
elif dataset == 'Imagenette':
    from DatasetImagenette import DatasetImagenette as Dataset
else:
    warn('Unknown dataset')
    exit
dataset_class = Dataset(param)
(x_train_array, y_train_array), (x_val_array, y_val_array) = dataset_class.getTrainVal()
num_train = x_train_array.shape[0]
num_labeled = x_train_array.shape[0]
neighbor_generator = NeighborGenerator(neighbor_noise_std, num_neighbor_per_anchor)
unlabeled_generator = UnlabeledGenerator(unlabeled_noise_std, num_unlabeled_per_labeled)
datahandler = DataHandler(dataset_class, 'cpu')
x_val, y_val = datahandler.loadValidation(x_val_array, y_val_array)
del x_val_array, y_val_array

logging = Logging(param.__dict__)

model = dataset_class.getModel()
criterion1 = CELoss()
criterion2 = GRLoss() if jacobian == 0 else JacobianReg()

# ---------------------------------------------------------------------------- #
#                             INSTANTIATE OPTIMIZER                            #
# ---------------------------------------------------------------------------- #

optimizer = Optimizers.Adam(model.parameters(), lr = param.lr, weight_decay=weight_decay)
steps_per_epoch = int(num_train * (num_unlabeled_per_labeled+1)/batch_size)
total_steps = num_epochs * steps_per_epoch
linear_steps = linear_epoch * steps_per_epoch
scheduler = lr_scheduler.LinearCosineLR(optimizer, total_steps, linear_steps = linear_steps)
starting_epoch = 1

# ---------------------------------------------------------------------------- #
#                      RESUME FROM CHECKPOINT IF REQUIRED                      #
# ---------------------------------------------------------------------------- #

if resume:
    # Read the result JSON file.
    if os.path.exists(json_log_path):
        checkpoint = torch.load(os.path.join(resume_dir, 'model_epoch={}.pt'.format(resume_epoch)), map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = resume_epoch + 1
        scheduler = lr_scheduler.LinearCosineLR(optimizer, total_steps, linear_steps=linear_steps, last_batch=resume_epoch*steps_per_epoch-1)
        success(f'Successfully loaded model file for epoch {resume_epoch}.')

# ---------------------------------------------------------------------------- #
#                                EPOCH TRAINING LOOP                           #
# ---------------------------------------------------------------------------- #
success(f'Starting{" adversarial" if adversarial else ""}{" intermediate noise injection" if inject_noise else ""} training {f"with" if grad_reg_lambda > 0 else " without"} {"Jacobian" if jacobian else "Gradient"} regularization.')

for epoch in range(starting_epoch, num_epochs + 1):

    print()
    info(f'Epoch {epoch}')

    ce_loss = 0.0
    gr_loss = 0.0
    train_acc = 0

    x_labeled_tensor = []
    y_labeled_tensor = []
    x_unlabeled_tensor = []
    y_unlabeled_tensor = []
    x_anchor_tensor = []
    y_anchor_tensor = []

    # Get augmented labeled data
    x_labeled_tensor, y_labeled_tensor = datahandler.loadAugmentedLabeled(x_train_array, y_train_array)

    # Generate unlabeled data
    if not inject_noise:
        x_unlabeled_tensor, y_unlabeled_tensor = unlabeled_generator.addUnlabeled(x_labeled_tensor)
        # Concatenate Labeled and Unlabeled data (Anchors)
        x_anchor_tensor = torch.cat((x_labeled_tensor, x_unlabeled_tensor), dim=0)
        y_anchor_tensor = torch.cat((y_labeled_tensor, y_unlabeled_tensor), dim=0)
    else:
        x_anchor_tensor = x_labeled_tensor
        y_anchor_tensor = y_labeled_tensor

    del x_labeled_tensor, y_labeled_tensor, x_unlabeled_tensor, y_unlabeled_tensor
    success_cnt = 0

    # Permute anchors
    idx = torch.randperm(y_anchor_tensor.shape[0], device='cpu')
    x_anchor_tensor = x_anchor_tensor[idx]
    y_anchor_tensor = y_anchor_tensor[idx]

    if inject_noise:
        x_anchor_tensor = torch.repeat_interleave(x_anchor_tensor, 2, dim=0)
        y_anchor_tensor = torch.repeat_interleave(y_anchor_tensor, 2, dim=0)

    # ---------------------------------------------------------------------------- #
    #                                BATCH TRAINING LOOP                           #
    # ---------------------------------------------------------------------------- #

    info('Training Loop')

    model.train()

    for batchIdx in trange(math.ceil(x_anchor_tensor.shape[0]/batch_size)):

        # Get training anchor batch.
        start_batch = batchIdx*batch_size
        end_batch = min((batchIdx+1)*batch_size, x_anchor_tensor.shape[0])

        x_anchor_clean = x_anchor_tensor[start_batch:end_batch].to(device)
        y_anchor_clean = y_anchor_tensor[start_batch:end_batch].to(device)

        if inject_noise:
            combined_x = torch.cat((x_anchor_clean, x_anchor_clean), dim=0) if not jacobian else x_anchor_clean

            if jacobian:
                combined_x.requires_grad = True
                x_anchor_clean_logits = model(combined_x, unlabeled_mode=batchIdx%2)
                gr = grad_reg_lambda * criterion2(combined_x, x_anchor_clean_logits)
            else:
                combined_logits, x_anchor_clean, x_neighbor_clean = model(combined_x, unlabeled_mode=batchIdx%2)
                x_anchor_clean_logits = combined_logits[0:batch_size]
                x_neighbor_clean_logits = combined_logits[batch_size]
                gr = grad_reg_lambda * criterion2(x_anchor_clean_logits.softmax(dim=-1), x_neighbor_clean_logits.softmax(dim=-1), x_anchor_clean, x_neighbor_clean)
            if not batchIdx % 2:
                ce = criterion1(x_anchor_clean_logits, y_anchor_clean)
            else:
                ce = torch.tensor([0.], device=device)
        else:
            x_neighbor_clean = neighbor_generator.addNeighbor(x_anchor_clean)

            if adversarial and (y_anchor_clean[0]!=-1):
                # Adversarial training

                # Get the ADVERSARIAL image(s) for this batch.

                # Squeeze to remove information about batch size.
                x_anchor_clean_squeezed = torch.squeeze(x_anchor_clean, 0)

                # Load the adversarial image(s).
                x_anchor_adv, y_anchor_adv, is_successful = fgsm(x_anchor_clean_squeezed, y_anchor_clean, model, epsilon, device)

                # Add the success to success_cnt.
                success_cnt += is_successful

                if jacobian:
                    # Adversarial training + Jacobian regularization
                    
                    # Jacobian regularization requires gradients to be preserved for calculation of Jacobian loss.
                    x_anchor_clean.requires_grad = True
                    x_anchor_adv.requires_grad = True
                    # Jacobian regularization does not require use of the neighbours.
                    # We pass clean img, adv img into the model.
                    combined_x = torch.cat((x_anchor_clean, x_anchor_adv), dim=0)
                
                    # Perform forward pass.
                    combined_logits = model(combined_x)

                    x_anchor_clean_logits_boundary = batch_size
                    x_anchor_adv_logits_boundary = x_anchor_clean_logits_boundary + batch_size

                    x_anchor_clean_logits = combined_logits[0:x_anchor_clean_logits_boundary]
                    x_anchor_adv_logits = combined_logits[x_anchor_clean_logits_boundary:x_anchor_adv_logits_boundary]

                    ce = (criterion1(x_anchor_clean_logits, y_anchor_clean) + adv_ratio * criterion1(x_anchor_adv_logits, y_anchor_clean)) / (1.0 + adv_ratio)
                    gr = grad_reg_lambda * criterion2(combined_x, combined_logits)

                else:
                    # Adversarial training + Our regularization

                    # Get the neighbors for the ADVERSARIAL image(s)
                    x_neighbor_adv = neighbor_generator.addNeighbor(x_anchor_adv)

                    # When doing adversarial training, we pass 4 things
                    # clean img, neighbor of clean img, adv img, neighbor of adv img.
                    combined_x = torch.cat((x_anchor_clean, x_neighbor_clean, x_anchor_adv, x_neighbor_adv), dim=0)

                    # Perform forward pass.
                    combined_logits = model(combined_x)

                    # Then, we get the logits for the respective images we passed in.
                    # clean img, neighbor of clean img, adv img, neighbor of adv img.
                    x_anchor_clean_logits_boundary = batch_size
                    x_neighbor_clean_logits_boundary = x_anchor_clean_logits_boundary + num_neighbor_per_anchor
                    x_anchor_adv_logits_boundary = x_neighbor_clean_logits_boundary + batch_size

                    x_anchor_clean_logits = combined_logits[0:x_anchor_clean_logits_boundary]
                    x_neighbor_clean_logits = combined_logits[x_anchor_clean_logits_boundary:x_neighbor_clean_logits_boundary]
                    x_anchor_adv_logits = combined_logits[x_neighbor_clean_logits_boundary:x_anchor_adv_logits_boundary]
                    x_neighbor_adv_logits = combined_logits[x_anchor_adv_logits_boundary:]

                    # Calculate individual losses.
                    ce = (criterion1(x_anchor_clean_logits, y_anchor_clean) + adv_ratio * criterion1(x_anchor_adv_logits, y_anchor_clean)) / (1.0 + adv_ratio)
                    gr1 = grad_reg_lambda * criterion2(x_anchor_clean_logits.softmax(dim=-1), x_neighbor_clean_logits.softmax(dim=-1), x_anchor_clean, x_neighbor_clean)
                    gr2 = grad_reg_lambda * criterion2(x_anchor_adv_logits.softmax(dim=-1), x_neighbor_adv_logits.softmax(dim=-1), x_anchor_adv, x_neighbor_adv)
                    gr = gr1 + gr2

            else:
                # No adversarial training

                if jacobian:
                    # No adversarial training + Jacobian regularization

                    # Jacobian Regularization requires gradients to be preserved for calculation of Jacobian loss.
                    x_anchor_clean.requires_grad = True
                    # Perform forward pass.
                    x_anchor_clean_logits = model(x_anchor_clean)

                    # Calculate losses based on Jacobian regularization.
                    ce = criterion1(x_anchor_clean_logits, y_anchor_clean)
                    gr = grad_reg_lambda * criterion2(x_anchor_clean, x_anchor_clean_logits)
                
                else:
                    # No adversarial training + Our regularization
                    # Note that if no adversarial training occurs, then we would only be concerned with:
                    # clean img, neighbor of clean img.
                    combined_x = torch.cat((x_anchor_clean, x_neighbor_clean), dim=0)

                    # Perform forward pass.
                    combined_logits = model(combined_x)

                    # Then, we get the logits for the respective images we passed in.
                    # Clean img, neighbor of clean img, adv img, neighbor of adv img.
                    x_anchor_clean_logits_boundary = batch_size

                    x_anchor_clean_logits = combined_logits[0:x_anchor_clean_logits_boundary]
                    x_neighbor_clean_logits = combined_logits[x_anchor_clean_logits_boundary:]

                    # Calculate individual losses.
                    ce = criterion1(x_anchor_clean_logits, y_anchor_clean)
                    gr = grad_reg_lambda * criterion2(x_anchor_clean_logits.softmax(dim=-1), x_neighbor_clean_logits.softmax(dim=-1), x_anchor_clean, x_neighbor_clean)

        # Calculate overall loss.
        ce_loss += ce.sum().item()
        gr_loss += gr.item()
        loss = ce + gr

        # Backprop and update model weights.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        # Calculate train loss and accuracy
        y_pred = x_anchor_clean_logits.softmax(dim=-1).argmax(-1)
        train_acc += torch.sum(y_anchor_clean == y_pred).cpu().item()

     # Clear up unused variables to reduce RAM usage.
    if jacobian:
        del x_anchor_clean, x_anchor_clean_logits, y_anchor_clean
    else:
        del x_anchor_clean, x_anchor_clean_logits, x_neighbor_clean, x_neighbor_clean_logits, y_anchor_clean

    train_acc /= num_labeled
    ce_loss /= num_labeled
    gr_loss /= batchIdx

    # ---------------------------------------------------------------------------- #
    #                           PER EPOCH VALIDATION LOOP                          #
    # ---------------------------------------------------------------------------- #

    info('Validation Loop')

    model.eval()
    val_batch = min(64, x_val.shape[0])
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batchIdx in trange(math.ceil(x_val.shape[0]/val_batch)):
            start_batch = batchIdx*val_batch
            end_batch = min((batchIdx+1)*val_batch,x_val.shape[0])
        
            x_anchor = x_val[start_batch:end_batch]
            y_anchor = y_val[start_batch:end_batch]
            
            model.eval()
            
            x_anchor = x_anchor.to(device)
            y_anchor = y_anchor.to(device)
            
            y_pred = model(x_anchor)
            loss = criterion1(y_pred, y_anchor)
            
            val_loss += loss.mean().item()
            y_pred = y_pred.softmax(-1).argmax(-1)
            val_acc += torch.sum(y_anchor == y_pred).cpu().item()

    val_loss /= batchIdx
    val_acc /= x_val.shape[0]

    del x_anchor, y_anchor
  
    # ---------------------------------------------------------------------------- #
    #                    LOGGING AND SAVING OF RESULTS                             #
    # ---------------------------------------------------------------------------- #

    info((f'Results for Epoch {epoch}:\n'
    f'CE Loss: {ce_loss}\n'
    f'GR Loss: {gr_loss}\n'
    f'Total Loss: {ce_loss+gr_loss}\n'
    f'Train Acc: {train_acc}\n\n'
    f'Val Loss: {val_loss}\n'
    f'Val Acc: {val_acc}')) 

    logging.writeLog(ce_loss, gr_loss, train_acc, val_loss, val_acc, epoch)

    if epoch % param.log_freq == 0:
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }
        logging.save_checkpoint(checkpoint_dict, f'model_epoch={epoch}.pt')

success('Code completed successfully.')
