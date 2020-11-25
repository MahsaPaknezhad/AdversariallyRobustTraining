#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utility file serving as the noise generator for unlabeled data and neighbor data.

import torch

class NoiseGenerator:
    """
    Add Gaussian noise to anchors point (labeled/unlabeled)
    to make neighbor points
    """
    def __init__(self, std, num_noise_per_anchor):
        """

        Parameters
        ----------
        mean : float
            Mean of Gaussian noise added.
        std : float
            Std of Gaussian noise added.
        num_neighbor : int
            Number of neighbor per anchor (labeled/unlabeled).
        device : 'cuda'/'cpu'
            Device to initiate tensor in.
        
        Returns
        -------
        None.

        """
        self.mean = 0
        self.std = std
        self.num_noise_per_anchor = num_noise_per_anchor
    
    def addNoise(self, x_anchor_tensor):
        """
        
        Parameters
        ----------
        x_anchor_tensor : Float tensor
        Shape: [num_anchor, C, H, W]
            Anchor point tensors (Labeled/Unlabeled).
        
        For example:
            [anchor1, anchor2]
    
        Returns
        -------
        x_neighbor_tensor : Float tensor
        Shape: [num_anchor * num_noise_per_anchor, C, H, W]    
            Noise tensors of anchor tensors (Labeled/Unlabeled).
            
        For example: (if num_noise_per_anchor is 3)
        x_noise_tensor = 
            [anchor_1_noise_1, anchor_1_noise_2, anchor_1_noise_3,
             anchor_2_noise_1, anchor_2_noise_2, anchor_2_noise_3]
        """
        
        device = 'cuda' if x_anchor_tensor.is_cuda else 'cpu'
        
        x_anchor_tensor_interleave = torch.repeat_interleave(x_anchor_tensor, self.num_noise_per_anchor, dim=0)
        #x = torch.tensor([1,4,6])
        #torch.repeat_interleave(x, 2, dim=0)
        #>tensor([1,1,4,4,6,6])
        
        x_anchor_tensor_interleave = x_anchor_tensor_interleave + torch.normal(self.mean, self.std, x_anchor_tensor_interleave.shape, device=device)
        
        return x_anchor_tensor_interleave
    
class NeighborGenerator(NoiseGenerator):
    def __init__(self, std, num_neighbor_per_anchor):
        super(NeighborGenerator, self).__init__(std, num_neighbor_per_anchor)
    
    def addNeighbor(self, x_anchor_tensor):
        """

        Parameters
        ----------
        x_anchor_tensor : Float Tensor
        Shape: [num_anchor, C, H, W]
            Anchor point tensors (Labeled/Unlabeled).
        
        Returns
        -------
        x_neighbor_tensor : Float tensor
        Shape: [num_anchor * num_neighbor, C, H, W]    
            Neighbor tensors of (Labeled/Unlabeled).

        """
        return self.addNoise(x_anchor_tensor)
    
    

class UnlabeledGenerator(NoiseGenerator):
    def __init__(self,std, num_unlabeled_per_labeled):
        super(UnlabeledGenerator, self).__init__(std, num_unlabeled_per_labeled)
    
    def addUnlabeled(self, x_labeled_tensor):
        """
        

        Parameters
        ----------
        x_labeled_tensor : Float Tensor
        Shape: [num_labeled, C, H, W]
            Labeled tensors.

        Returns
        -------
        x_unlabeled_tensor : Float Tensor
        Shape: [num_labeled * num_unlabeled_per_labeled, C, H, W]
            Unlabeled tensors.
        
        y_unlabeled_tensor : Long Tensor
        Shape: [num_labeled * num_unlabeled_per_labeled, ]
            A Long tensor full of -1
            As we set the label of unlabeled points to be -1.

        """
        device = 'cuda' if x_labeled_tensor.is_cuda else 'cpu'
        
        x_unlabeled_tensor = self.addNoise(x_labeled_tensor)
        
        y_unlabeled_tensor = torch.ones((x_unlabeled_tensor.shape[0],), dtype=torch.long, device=device)
        #The label of unlabeled point is set to -1 to differentiate
        y_unlabeled_tensor.fill_(-1)
        
        return x_unlabeled_tensor, y_unlabeled_tensor