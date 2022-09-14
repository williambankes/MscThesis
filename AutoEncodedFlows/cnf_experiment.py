# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:03:26 2022

@author: William
"""

from torchdyn.core import NeuralODE
from torchdyn.nn import Augmenter
from torchdyn.models import CNF, hutch_trace
from AutoEncodedFlows.datasets import TwoMoonDataset

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.distributions import MultivariateNormal
import pytorch_lightning as pl

import numpy as np

torch.set_num_threads(16)
torch.manual_seed(1)

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


if __name__ == '__main__':

    dims, hidden_dim = 2, 64
    vector_field = nn.Sequential(
                nn.Linear(dims, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim , hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, dims))
    
    trainloader = data.DataLoader(TwoMoonDataset(n_samples=1<<14, noise=0.07),
                                  batch_size=1024, shuffle=True)
        
    data_points = list()
    
    for _ in range(10):
        
        learn = CNFLearner(vector_field, 2)
        trainer = pl.Trainer(gpus=1, min_epochs=400, max_epochs=600)
        trainer.fit(learn, train_dataloaders=trainloader)
        
        data_points.append(learn.losses[-1].detach().numpy())
    
        torch.cuda.empty_cache()
        del learn, trainer
        
    print(np.mean(data_points))
    print(np.std(data_points))