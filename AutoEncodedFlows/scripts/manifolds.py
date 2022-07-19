# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:12:10 2022

@author: William
"""
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torchdyn.core import NeuralODE
from torchdyn.models import CNF, hutch_trace
from torchdyn.nn import Augmenter
import pytorch_lightning as pl
from AutoEncodedFlows.utils.experiments import Experiment
from AutoEncodedFlows.datasets import Manifold1DDataset



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

class VectorFieldNoTime(nn.Module):
    
    def __init__(self, dims, hidden_dims):
        
        super().__init__()
        
        self.__name__ = 'cnf_vector_field'
        self.network = nn.Sequential(
                           nn.Linear(dims, hidden_dims),
                           nn.Tanh(),
                           nn.Linear(hidden_dims, hidden_dims),
                           nn.Tanh(),
                           nn.Linear(hidden_dims, hidden_dims),
                           nn.Tanh(),
                           nn.Linear(hidden_dims, dims))
    
    def forward(self, x):
        return self.network(x)
    
class VectorFieldTime(nn.Module):
    pass


if __name__ == '__main__':
                
    trainer_args = {'gpus':1 if torch.cuda.is_available() else 0,
                    'min_epochs':1,
                    'max_epochs':1,
                    'enable_checkpointing':False}
    learner_args = {'dims':2}
    model_args = {'dims':2,
                  'hidden_dims':64}
    dataset_args = {'n_samples':10_000}
    dataloader_args = {'batch_size':508,
                       'shuffle':True}
        
    exp1 = Experiment(project='AutoEncodingFlows',
                      tags=['MscThesis', 'AutoEncoder'],
                      learner=CNFLearner,
                      model=VectorFieldNoTime,
                      dataset=Manifold1DDataset,
                      trainer_args=trainer_args,
                      learner_args=learner_args,
                      model_args=model_args,
                      dataset_args=dataset_args,
                      dataloader_args=dataloader_args)

    try:
        exp1.run()
    finally:
        exp1.finish()
    