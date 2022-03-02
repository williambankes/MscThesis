# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:54:54 2022

@author: William Bankes
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_density_contours
from transforms import AffineTransform, PlanarTransform
from normalising_flows import NormalisingFlow, CompositeFlow


#%%

#We first define a multi-modal density function:

def density_func_1(x):
    
    term1 = 0.5*((torch.norm(x, p=2, dim=1)-2)/0.4)**2
    
    inner_term1 = -0.5 * ((x[:,0] - 2)/0.6)**2
    inner_term2 = -0.5 * ((x[:,0] + 2)/0.6)**2
    
    term2 = inner_term1.exp() + inner_term2.exp()
    
    return (- term1 + term2.log()).exp()


plot_density_contours(density_func_1, 'dense func 1')

#%%
#Define a composite transform and normalising flow:
#Define a base distribution:
z_mean = torch.zeros(2)
z_sigma = torch.eye(2)
G = dist.MultivariateNormal(loc=z_mean, covariance_matrix=z_sigma)
    
#Define the Composite flow:
tf = CompositeFlow(dims=2, Transform=PlanarTransform, num=8)
nf = NormalisingFlow(2, tf, G)

loss = nf.reverse_KL(density_func_1, epochs=1000)

#Plot loss of the model training:
fig, axs = plt.subplots(figsize=(10,7))
axs.plot(loss)
axs.set_title('training loss')

#plot the samples:
    
samp = nf.reverse_sample(n_samples=1000)
    
fig, axs = plt.subplots()
axs.scatter(samp[:,0], samp[:,1])


