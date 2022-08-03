# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:51:03 2022

@author: William
"""

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torchdyn.core import NeuralODE
from torchdyn.models import CNF, hutch_trace
from torchdyn.nn import Augmenter, DepthCat
import pytorch_lightning as pl
from AutoEncodedFlows.modules import MADE
from typing import Union, Callable

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
        self.losses = list()
        
        #Define model parameters:
        ode_solver_args = {'solver':'tsit5'}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #Create Jacobian noise dist and base dist:
        self.mean = torch.zeros(self.dims).to(device)
        self.cov = torch.eye(self.dims).to(device)
        self.base_dist = MultivariateNormal(self.mean, self.cov)
        
        #Create model:
        cnf = CNF(vector_field,
                  noise_dist=self.base_dist,
                  trace_estimator=hutch_trace)
        node = NeuralODE(cnf, **ode_solver_args)
        self.model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1),
                                   node)
        wandb.watch(self.model)
        
        
    def forward(self, x):
        
        #Set model time span forward:
        self.model[1].t_span = torch.linspace(0, 1, 10)
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        self.iters += 1 
        t_eval, xtrJ = self.model(batch)
        xtrJ = xtrJ[-1] #select the end point of the trajectory:
        logprob = self.base_dist.log_prob(xtrJ[:,1:]).to(batch) - xtrJ[:,0]
        loss = -torch.mean(logprob)
        
        if self.current_epoch % 10 == 0:
            self.losses.append(loss.cpu().detach())
        
        #wandb logging:
        wandb.log({'training loss': loss.detach().item(),
                   'epoch': self.current_epoch})
           
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
    
    def __init__(self, dims, hidden_dims):
        
        super().__init__()
        
        self.__name__ ='cnf_vector_field_w_time'
        self.network = nn.Sequential(
                           DepthCat(1),
                           nn.Linear(dims + 1, hidden_dims),
                           nn.Tanh(),
                           DepthCat(1),
                           nn.Linear(hidden_dims + 1, hidden_dims),
                           nn.Tanh(),
                           DepthCat(1),
                           nn.Linear(hidden_dims + 1, hidden_dims),
                           nn.Tanh(),
                           nn.Linear(hidden_dims, dims))
        
    def forward(self, x):
        return self.network(x)


class VectorFieldMasked(nn.Module):
    
    def __init__(self, dims, hidden_dims):
        """
        Taken from the paper: https://proceedings.mlr.press/v139/bilos21a.html
        as a means of explicitly controlling the trace of the jacobian.    

        Parameters
        ----------
        dims : int
            Input Dimension.
        hidden_dims : int
            Dimension of hidden layers.

        Returns
        -------
        None.

        """
        
    
        super().__init__()
        
        #Define point wise network:
        self.h_net = nn.Sequential(nn.Linear(dims, hidden_dims),
                                   nn.Tanh(),
                                   nn.Linear(hidden_dims, hidden_dims),
                                   nn.Tanh(),
                                   nn.Linear(hidden_dims, hidden_dims),
                                   nn.Tanh(),
                                   nn.Linear(hidden_dims, dims))
            
        #Define autoregressive component:
        self.MADE_net_1 = MADE(dims, [hidden_dims]*2, dims, natural_ordering=False)
        self.MADE_net_2 = MADE(dims, [hidden_dims]*2, dims, natural_ordering=True)
                    
            
        #Define scale parameters:
        self.params = nn.parameter.Parameter(torch.randn(1, dims))    
        
    def forward(self, x):
        
        h_net_output = self.h_net(x)
        pointwise_element = - h_net_output + h_net_output.sum(0)
        inner_element = self.MADE_net_1(x) + self.MADE_net_2(x)
        trace_element = self.params * x
        
        return pointwise_element + inner_element + trace_element

 
def MaskedTrace(model):
    return model.params.sum(0)

class MaskedCNF(nn.Module):
    def __init__(self, net:nn.Module, trace_estimator:Union[Callable, None]=None, noise_dist=None, order=1):
        """Continuous Normalizing Flow
        :param net: function parametrizing the datasets vector field.
        :type net: nn.Module
        :param trace_estimator: specifies the strategy to otbain Jacobian traces. Options: (autograd_trace, hutch_trace)
        :type trace_estimator: Callable
        :param noise_dist: distribution of noise vectors sampled for stochastic trace estimators. Needs to have a `.sample` method.
        :type noise_dist: torch.distributions.Distribution
        :param order: specifies parameters of the Neural DE.
        :type order: int
        """
        super().__init__()
        self.net, self.order = net, order # order at the CNF level will be merged with DEFunc
        self.trace_estimator = MaskedTrace

    def forward(self, x):
        with torch.set_grad_enabled(True):
            # first dimension is reserved to divergence propagation
            x_in = x[:,1:].requires_grad_(True)

            # the neural network will handle the datasets-dynamics here
            if self.order > 1: x_out = self.higher_order(x_in)
            else: x_out = self.net(x_in)

            trJ = self.trace_estimator(self.net)
        return torch.cat([-trJ[:, None], x_out], 1) + 0*x 


if __name__ == '__main__':

    #Move to test folder at some point:
    data = torch.randn(10,2)

    model = VectorFieldMasked(2,8)

    output = model(data)  

    jac = torch.autograd.functional.jacobian(model, data)
    print('jacobian output\n', output)
    print('jacobian pre train\n', jac[1,:,1,:].diag())      
    print('trace element\n', model.params)
        
        
        
        