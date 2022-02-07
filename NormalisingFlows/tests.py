# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:24:50 2022

@author: William Bankes
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_density_contours
from transforms import AffineTransform
from normalising_flows import NormalisingFlow


#%%
class TestFlow(nn.Module):
    
    def __init__(self, dims):
        super(TestFlow, self).__init__()
        
        self.dims = dims
        self.flow1 = AffineTransform(dims)
        
    def forward(self, x):
        
        z, log_detJ = self.flow1(x)
                
        return z, log_detJ 



#Test:    
target_mean = torch.ones([1,2])
target_sigma = torch.eye(2) * 4
target_dist = dist.MultivariateNormal(loc=target_mean,
                                      covariance_matrix=target_sigma)

N = 1000
data = target_dist.sample(torch.Size([N]))
data = data.reshape(N, 2)

print('data shape:', data.shape)

z_mean = torch.zeros(2)
z_sigma = torch.eye(2)
G = dist.MultivariateNormal(loc=z_mean, covariance_matrix=z_sigma)
    
nf = NormalisingFlow(2, TestFlow, G)

out = nf.forward_KL(data, 5000)

# Plot losses
fig, axs = plt.subplots(figsize=(10,7))
axs.plot(out)
axs.set_title('training loss')

# Plot actual and 'found' distributions:
plot_density_contours(lambda x: np.exp(nf.density_estimation_forward(x)) , 'backward')
plot_density_contours(lambda x: target_dist.log_prob(x).exp(), 'target')

#print optimised params and actual:
a = nf.transform.flow1.alpha
print('Alpha param:', torch.linalg.inv(a@a.T + 0.01*torch.eye(2)))