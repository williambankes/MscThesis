# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:34:50 2022

@author: William
"""

#%% Imports

from torchdyn.core import NeuralODE
from torchdyn.nn import Augmenter, DepthCat
from torchdyn.models import CNF, hutch_trace
from torchdyn.utils import plot_2D_state_space, plot_2D_depth_trajectory, plot_2D_space_depth
from AutoEncodedFlows.datasets import TwoMoonDataset, Manifold1DDataset
from AutoEncodedFlows.models import GradientNetwork, PrintLayer

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.distributions import MultivariateNormal
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt

torch.set_num_threads(16)
torch.manual_seed(1)
#%% Create pytorch lightning learner:
        
class CNFLearner(pl.LightningModule):
    
    def __init__(self, vector_field:nn.Module, dims:int):
        """
        Learner setup for Torchdyn CNF model

        Parameters
        ----------
        model : nn.Module
            AutoEncoder model with encode and decode methods

        Returns
        -------
        None.

        """
        
        super().__init__()
        self.__name__ = 'CNFLearner'
        self.iters = 0
        self.dims = dims
        
        ode_solver_args = {'solver':'tsit5'}
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.mean = torch.zeros(self.dims).to(device)
        self.cov = torch.eye(self.dims).to(device)
        self.base_dist = MultivariateNormal(self.mean, self.cov)
        
        cnf = CNF(vector_field,
                  noise_dist=self.base_dist,
                  trace_estimator=hutch_trace)
        node = NeuralODE(cnf, **ode_solver_args)
        self.model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1),
                                   node)
        self.losses = list()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.iters += 1 
        t_eval, xtrJ = self.model(batch)
        xtrJ = xtrJ[-1] #select the end point of the trajectory:
        logprob = self.base_dist.log_prob(xtrJ[:,1:]).to(batch) - xtrJ[:,0]
        loss = -torch.mean(logprob)
        
        if self.current_epoch % 10 == 0:
            self.losses.append(loss.cpu().detach())
        
        return {'loss': loss}   
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=2e-3, weight_decay=1e-5)


#%% Define model and run CNF:
dims, hidden_dim = 2, 64
vector_field = nn.Sequential(
            nn.Linear(dims, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim , hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dims))

vector_field3 = GradientNetwork(dims=2, time_grad=False, hidden_dim=64)
trainloader = data.DataLoader(TwoMoonDataset(n_samples=1<<14, noise=0.07),
                              batch_size=1024, shuffle=True)
    
learn = CNFLearner(vector_field, 2)
trainer = pl.Trainer(gpus=1, min_epochs=400, max_epochs=600)
trainer.fit(learn, train_dataloaders=trainloader)

#%% Visualise the sample:
torch.cuda.empty_cache() 
sample = learn.base_dist.sample(torch.Size([1000])).cuda()
trainer.model.model[1].t_span = torch.linspace(1, 0, 2)
trainer.model.cuda()
_, new_x = trainer.model.model(sample)
new_x = new_x.cpu().detach()[-1,:,1:]

#matplotlib vis:
fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

data_points = TwoMoonDataset(n_samples=1000, noise=0.07)
axs[0].scatter(data_points[:,0], data_points[:,1])

axs[1].scatter(new_x[:,0], new_x[:,1])

axs[2].plot(trainer.model.losses)

#%% 1D Manifold Experiment:
    
dims, hidden_dim = 2, 64
vector_field = nn.Sequential(
            nn.Linear(dims, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim , hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dims))

vector_field3 = GradientNetwork(dims=2, time_grad=False, hidden_dim=64)
trainloader = data.DataLoader(Manifold1DDataset(n_samples=10_000, noise=0.07),
                              batch_size=1024, shuffle=True)
    
learn = CNFLearner(vector_field, 2)
trainer = pl.Trainer(gpus=1, min_epochs=400, max_epochs=600)
trainer.fit(learn, train_dataloaders=trainloader)


#%% Visualise the results:
torch.cuda.empty_cache()
N = 1000 
sample = learn.base_dist.sample(torch.Size([N])).cuda()
trainer.model.model[1].t_span = torch.linspace(1, 0, 10)
trainer.model.cuda()
_, traj = trainer.model.model(sample)
new_x = traj.cpu().detach()[-1,:,1:]

data_points = Manifold1DDataset(n_samples=1000, noise=0.07)
man_dataset = data_points.get_dataset().cuda()
trainer.model.model[1].t_span = torch.linspace(0, 1, 10)
_, traj2 = trainer.model.model(man_dataset)
new_y = traj2.cpu().detach()[-1,:,1:]

#matplotlib vis:
fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

axs[0].scatter(data_points[:,0], data_points[:,1])
axs[0].set_title('Original Data')

axs[1].scatter(new_x[:,0], new_x[:,1])
axs[1].set_title('Samples to Distribution')

axs[2].scatter(new_y[:,0], new_y[:,1])
axs[2].set_title('Data to Distribution')

fig, axs = plt.subplots()
axs.plot(trainer.model.losses)

#Plot the depth perception:
#Generate random indices:
indices = np.random.randint(low=0, high=traj.shape[1], size=100)
plot_2D_depth_trajectory(np.linspace(1,0,10),
                         traj[:,indices,1:].cpu().detach(),
                         np.ones(100), 100)

plot_2D_state_space(traj[:,:,1:].cpu().detach(),
                    np.ones(N), N)

plot_2D_space_depth(torch.linspace(1, 0, 10),
                    traj[:,:,1:].cpu().detach(),
                    torch.ones(N), N)


