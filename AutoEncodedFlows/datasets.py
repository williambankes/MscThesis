# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:48:15 2022

@author: William
"""

import torch
from torch.utils import data

from sklearn import datasets

class SCurveDataset(data.Dataset):
    
    def __init__(self, n_samples, noise=0.0):
        
        self.n_samples = n_samples
        self.data = datasets.make_s_curve(n_samples, noise=0.0)[0]
        self.data = torch.tensor(self.data).float()
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    