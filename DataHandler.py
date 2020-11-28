#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class DataHandler:
    def __init__(self, dataset_class, device):
        self.transform_train = dataset_class.transform_train
        self.transform_val = dataset_class.transform_val
        self.transform_test = dataset_class.transform_test
        self.device = device


    def _loadNumpyToTensor(self, x_array, y_array, transform):

        """

        Parameters
        ----------
        x_array : string
            Numpy filename of data input.
        y_array : string
            Numpy filename of data label.
        transform : Pytorch Transformation.Compose
            transformation to be applied on.

        Returns
        -------
        x_tensor : Float Tensor
        Shape [N, C, H, W].
        y_tensor : Long Tensor
        Shape [N, C, H, W].

        """

        epoch_dataset = CustomDataset(x_array, y_array, transform)

        epoch_loader = torch.utils.data.DataLoader(epoch_dataset, batch_size=x_array.shape[0], shuffle=True, num_workers=4, pin_memory=True if self.device == 'cuda' else False)

        for data in epoch_loader:
            x_tensor, y_tensor = data
            x_tensor = x_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)

        return x_tensor, y_tensor

    def loadValidation(self, x_val_array, y_val_array):
        return self._loadNumpyToTensor(x_val_array, y_val_array, self.transform_val)

    def loadTest(self, x_test_array, y_test_array):
        return self._loadNumpyToTensor(x_test_array, y_test_array, self.transform_test)

    def loadAugmentedLabeled(self, x_train_array, y_train_array):
        device = self.device

        x_labeled_tensor, y_labeled_tensor = self._loadNumpyToTensor(x_train_array, y_train_array, self.transform_train)

        x_labeled_tensor = x_labeled_tensor.to(device)
        y_labeled_tensor = y_labeled_tensor.to(device)

        return x_labeled_tensor, y_labeled_tensor

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.x_data[index], self.y_data[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.y_data.shape[0]
