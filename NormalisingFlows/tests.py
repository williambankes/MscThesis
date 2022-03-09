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

from NormalisingFlows.utils import plot_density_contours, plot_density_image
from NormalisingFlows.transforms import AffineTransform, PlanarTransform
from NormalisingFlows.normalising_flows import NormalisingFlow, CompositeFlow

from sklearn.datasets import make_moons, make_classification
  
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
z_mean = torch.zeros(2)
z_sigma = torch.eye(2)
G = dist.MultivariateNormal(loc=z_mean, covariance_matrix=z_sigma)
    
#Generate moon data:    
moon_data, label = make_moons(n_samples=2000, noise=0.01)
moon_data = moon_data * 2

#Shift slightly to the left:
moon_data[:,0] -= 1

plt.scatter(moon_data[:, 0], moon_data[:, 1])
plt.ylim([-5, 5])
plt.xlim([-5, 5])

torch_data = torch.tensor(moon_data, dtype=torch.float)
    

#%%
#Look into non-contour methods of plotting these outputs...

num = 8

flow = CompositeFlow(dims=2, Transform=PlanarTransform, num=num)
nf = NormalisingFlow(2, flow, G)
out = nf.forward_KL(torch_data, 5000)

plot_density_contours(lambda x: np.exp(nf.density_estimation_forward(x)),
                      '{} planar transforms'.format(num))

#%%
plot_density_image(lambda x: np.exp(nf.density_estimation_forward(x)), 'test')


#%%
for n in [5,10,15,20,25,32]:
    tf2 = CompositeFlow(dims=2, Transform=PlanarTransform, num=n)
    nf3 = NormalisingFlow(2, tf2, G)
    out = nf3.forward_KL(torch_data, 5000)
    
    plot_density_contours(lambda x: np.exp(nf3.density_estimation_forward(x)),
                          '{} planar transforms'.format(n))
    plot_density_image(lambda x: np.exp(nf3.density_estimation_forward(x)),
                          '{} planar transforms'.format(n))
    
#%%
#Sample from two Gaussians
G1 = dist.MultivariateNormal(loc=torch.tensor([1.,1.]),
                             covariance_matrix=torch.eye(2)*0.2)
G2 = dist.MultivariateNormal(loc=torch.tensor([-1.,-1.]),
                             covariance_matrix=torch.eye(2)*0.2)

samples = torch.concat([G1.sample((50,)),
                        G2.sample((50,))])

plt.scatter(samples[:,0], samples[:,1])

#%%

flow = CompositeFlow(dims=2, Transform=PlanarTransform, num=8)
nf = NormalisingFlow(2, flow, G)
out = nf.forward_KL(samples, 3000)

#%%
plot_density_contours(lambda x: np.exp(nf.density_estimation_forward(x)),
                      '{} planar transforms'.format(5))

#%%
#forward density estimation of Normalising flow:    
out = plot_density_contours(lambda x: G.log_prob(x).exp(), 'target')

#G1 after one normalising flow...
flow = CompositeFlow(dims=2, Transform=PlanarTransform, num=1)
nf = NormalisingFlow(2, flow, G)

#Re-create planar flow hexbins:
plot_density_contours(lambda x: np.exp(nf.density_estimation_forward(x)),
                      '{} planar transforms'.format(5))

#%%
z_mean = torch.zeros(2)
z_sigma = torch.eye(2)
G = dist.MultivariateNormal(loc=z_mean, covariance_matrix=z_sigma)
gsample = G.sample((5000,))

fig, axs = plt.subplots()
axs.hexbin(gsample[:,0], gsample[:,1], C=G.log_prob(gsample).exp(), cmap='rainbow')

flow = CompositeFlow(dims=2, Transform=PlanarTransform, num=1)
flow.flows[0].w = nn.parameter.Parameter(torch.tensor([[5.,1.]]).T)
flow.flows[0].v = flow.flows[0].w
nf = NormalisingFlow(2, flow, G)

sample, sample_prob, detJ = nf.reverse_sample(5000)

fig, axs = plt.subplots()
axs.hexbin(sample[:,0], sample[:,1], C=sample_prob.exp(), cmap='rainbow')

#p(x) = p(u)|J_T|^(-1)
sample_prob_new = sample_prob - detJ.detach().reshape(-1)

fig, axs = plt.subplots()
axs.hexbin(sample[:,0], sample[:,1], C=sample_prob_new.exp(),
           cmap='rainbow')

#%%

a = torch.tensor([[0.,0.],[-1.5,1.]])

out = flow(a)
print(out)












     


    