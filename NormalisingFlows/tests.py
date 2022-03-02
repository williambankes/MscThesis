# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:24:50 2022

To do:
    
- Test on moon data 
    -> understand stability of training flows
    -> reproduce pyMC3 tutorial to some degree (although reverse KL used there)

- Develop TestFlow modules for better handling of forward and logdet
    -> input multiple layer instructions at once for easier coding /iteration
    
- Understand how expressivity of planar flows works
- Implement low dim noise model



@author: William Bankes
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_density_contours
from transforms import AffineTransform, PlanarTransform2
from normalising_flows import NormalisingFlow

from sklearn.datasets import make_moons, make_classification


#%%
class TestFlow(nn.Module):
    
    def __init__(self, dims):
        super(TestFlow, self).__init__()
        
        self.dims = dims
        #self.flow1 = AffineTransform(dims)
        self.flow1 = PlanarTransform(dims)
        self.flow2 = PlanarTransform(dims)
        self.flow3 = PlanarTransform(dims)
        self.flow4 = PlanarTransform(dims)
        self.flow5 = PlanarTransform(dims)

        
    def forward(self, x):
        
        z, log_detJ = self.flow1(x)        
        z, log_detJ2 = self.flow2(z)
        z, log_detJ3 = self.flow3(z)
        z, log_detJ4 = self.flow4(z)
        z, log_detJ5 = self.flow5(z)
                
        return z, log_detJ + log_detJ2 + log_detJ3 + log_detJ4 + log_detJ5
    
class TestFlow2(nn.Module):
    
    def __init__(self, dims, Transform, num):
        
        super().__init__()
        
        self.dims = dims
        self.num = num
        self.flows = nn.ModuleList()
        
        for i in range(num):
            self.flows.append(Transform(dims))

    def forward(self, z):
        
        logdet = torch.zeros([z.shape[0], 1])
        
        for i in range(self.num):
            z, logdetT = self.flows[i](z)
            
            logdet += logdetT
            
        return z, logdet


#%%
#Test 1:    
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
    
nf = NormalisingFlow(2, TestFlow(2), G)
out = nf.forward_KL(data, 5000)

# Plot losses
fig, axs = plt.subplots(figsize=(10,7))
axs.plot(out)
axs.set_title('training loss')

# Plot actual and 'found' distributions:
plot_density_contours(lambda x: np.exp(nf.density_estimation_forward(x)) , 'backward')
plot_density_contours(lambda x: target_dist.log_prob(x).exp(), 'target')

#print optimised params and actual:
#a = nf.transform.flow1.alpha
#print('Alpha param:', torch.linalg.inv(a@a.T + 0.01*torch.eye(2)))

#%%
#Test 2:
    
#Generate moon data:
moon_data, label = make_moons(n_samples=2000, noise=0.1)
moon_data = moon_data * 2
plt.scatter(moon_data[:, 0], moon_data[:, 1])
plt.ylim([-5, 5])
plt.xlim([-5, 5])

torch_data = torch.tensor(moon_data, dtype=torch.float)
    

#%%
tf2 = TestFlow2(dims=2, Transform=PlanarTransform2, num=32)
nf3 = NormalisingFlow(2, tf2, G)
out = nf3.forward_KL(torch_data, 5000)

plot_density_contours(lambda x: np.exp(nf3.density_estimation_forward(x)),
                      '{} planar transforms'.format(32))

#%%
#Test flow 2

for n in [5,10,15,20,25,32]:
    tf2 = TestFlow2(dims=2, Transform=PlanarTransform2, num=n)
    nf3 = NormalisingFlow(2, tf2, G)
    out = nf3.forward_KL(torch_data, 5000)
    
    plot_density_contours(lambda x: np.exp(nf3.density_estimation_forward(x)),
                          '{} planar transforms'.format(n))
    
    
#%%
#Try fitting to multimodal gaussian first... then fit to two moons? -> some meta learning to improve stability?
#Define samples from the multi-modal Gaussian:
    

    
    
    
    
    
    
    

