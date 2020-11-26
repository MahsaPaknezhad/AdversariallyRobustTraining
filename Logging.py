#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility class that handles printing to the console and logging results of training.

import os
import json
import datetime
import torch

from torch.utils.tensorboard import SummaryWriter

class Logging:
    def __init__(self, result_dict):
        # If not resuming training
        if 'ce_loss' not in result_dict:
            result_dict['ce_loss'] = []
            result_dict['gr_loss'] = []
            result_dict['train_acc'] = []
            result_dict['val_loss'] = []
            result_dict['val_acc'] = []
            
            # Result directory
            if result_dict['setting'] != '':
                result_dict['result_dir'] = os.path.join(result_dict["output_dir"],
                                                result_dict["dataset"],
                                                result_dict["setting"],
                                                f'Seed_{result_dict["seed"]}')
            else:
                result_dict['result_dir'] = os.path.join(result_dict["output_dir"],
                                        result_dict["dataset"],
                                        f'num_train={result_dict["num_train"]}&lambda={result_dict["grad_reg_lambda"]}&unlabeled={result_dict["num_unlabeled_per_labeled"]}&neighbor{result_dict["num_neighbor_per_anchor"]}&adversarial={result_dict["adversarial"]}&epoch={result_dict["num_epochs"]}',
                                        f'Seed_{result_dict["seed"]}')

        # Tensorboard writer
        self.writer = SummaryWriter(os.path.join(result_dict['result_dir'], 'tensorboard'))
        self.result_dict = result_dict
        
    def writeLog(self, ce_loss, gr_loss, train_acc, val_loss, val_acc, epoch):
        self.result_dict['ce_loss'].append(ce_loss)
        self.result_dict['gr_loss'].append(gr_loss)
        self.result_dict['train_acc'].append(train_acc)
        self.result_dict['val_loss'].append(val_loss)
        self.result_dict['val_acc'].append(val_acc)
        
        # Save to json file
        with open(os.path.join(self.result_dict['result_dir'],'result.json'), 'w') as file:
            json.dump(self.result_dict, file, indent=4)
            
        # Save to tensorboard
        self.writer.add_scalar('Train/ce_loss', ce_loss, epoch)
        self.writer.add_scalar('Train/gr_loss', gr_loss, epoch)
        self.writer.add_scalar('Train/train_acc', train_acc, epoch)
        
        self.writer.add_scalar('Val/val_loss', val_loss, epoch)
        self.writer.add_scalar('Val/val_acc', val_acc, epoch)

    def save_checkpoint(self, state, filename):
        target_path = os.path.join(self.result_dict['result_dir'],filename)
        torch.save(state, target_path)
        success(f'Model saved to {target_path}')
        
def info(log_str):
    print(f'\033[96m[INFO] {log_str}\033[0m')

def success(log_str):
    print(f'\033[92m[SUCCESS] {log_str}\033[0m')

def warn(log_str):
    print(f'\033[93m\033[1m[WARN] {log_str}\033[0m')

