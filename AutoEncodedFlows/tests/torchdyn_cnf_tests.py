# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:34:50 2022

@author: William
"""

#%% Imports

from torchdyn.core import NeuralODE
from torchdyn.nn import Augmenter, DepthCat
from torchdyn.models import CNF, hutch_trace
from AutoEncodedFlows.datasets import TwoMoonDataset
from AutoEncodedFlows.models import GradientNetwork, PrintLayer

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.distributions import MultivariateNormal
import pytorch_lightning as pl

import matplotlib.pyplot as plt

torch.set_num_threads(16)
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
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.iters += 1 
        t_eval, xtrJ = self.model(batch)
        xtrJ = xtrJ[-1] #select the end point of the trajectory:
        logprob = self.base_dist.log_prob(xtrJ[:,1:]).to(batch) - xtrJ[:,0]
        loss = -torch.mean(logprob)
        return {'loss': loss}   
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0001)


#%% Define model and run CNF:
dims, hidden_dim = 2, 64
vector_field = nn.Sequential(
            DepthCat(1),
            nn.Linear(dims + 1, hidden_dim),
            nn.Tanh(),
            DepthCat(1),
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            DepthCat(1),
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dims))

vector_field3 = GradientNetwork(dims=2, time_grad=False, hidden_dim=64)
trainloader = data.DataLoader(TwoMoonDataset(n_samples=1000, noise=0.07),
                              batch_size=256, shuffle=True)
    
learn = CNFLearner(vector_field, 2)
trainer = pl.Trainer(gpus=1, max_epochs=1000)
trainer.fit(learn, train_dataloaders=trainloader)

#%% Visualise the sample:
torch.cuda.empty_cache() 
sample = learn.base_dist.sample(torch.Size([1000])).cuda()
trainer.model.model[1].s_span = torch.linspace(1, 0, 2)
trainer.model.cuda()
_, new_x = trainer.model.model(sample)
new_x = new_x.cpu().detach()[-1,:,1:]

#matplotlib vis:
fig, axs = plt.subplots()
axs.scatter(new_x[:,0], new_x[:,1])

#%%

fig, axs = plt.subplots()
data = TwoMoonDataset(n_samples=1000, noise=0.07)
axs.scatter(data[:,0], data[:,1])
















