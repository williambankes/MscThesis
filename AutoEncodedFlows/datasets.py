# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:48:15 2022

@author: William
"""

import torch
from torch.utils import data

from sklearn import datasets

class SCurveDataset(data.Dataset):
    
    def __init__(self, n_samples, extra_dims=0, noise=0.0):
        
        self.n_samples = n_samples
        self.data, self.labels = datasets.make_s_curve(n_samples, noise=0.0)
        self.data = torch.tensor(self.data).float()
        self.labels = torch.tensor(self.labels).float()
                  
        self.data = self.data + torch.tensor([0., -1., 0.])
        
        #extra dimensionality:
        zeros = torch.zeros((self.data.shape[0], extra_dims))
        self.data = torch.concat([self.data, zeros], axis=-1)
        self.extra_dims=extra_dims
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
    def get_dataset(self, colours=False):
        return self.data, self.labels
    
class TwoMoonDataset(data.Dataset):
    
    def __init__(self, n_samples, extra_dims=0, noise=0.0):
        
        self.n_samples = n_samples
        self.data, self.labels = datasets.make_moons(n_samples, noise=noise)
        self.data = torch.tensor(self.data).float()
        self.labels = torch.tensor(self.labels).float()
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_dataset(self):
        return self.data, self.labels
        
    