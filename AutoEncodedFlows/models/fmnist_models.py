# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 14:34:37 2022

@author: William
"""

import torch
import torch.nn as nn
from torchdyn.nn import DepthCat

from AutoEncodedFlows.models.modules import NeuralODEWrapper, Projection1D
from AutoEncodedFlows.models.modules import SequentialFlow

class AENODEConvModel(nn.Module):
    
    def __init__(self, kernel:int, ode_solver_args=None):
 
        super().__init__()
        
        assert kernel % 2 != 0, 'kernel must be odd valued int'
        padding = int(0.5*(kernel - 1))
        kernel = int(kernel)
        
        if ode_solver_args is None: ode_solver_args = {'solver':'dopri5'}
               
        
        encoder_net_data = nn.Sequential(nn.Conv2d(in_channels=1,
                                                       out_channels=4,
                                                       kernel_size=kernel,
                                                       padding=padding),
                                        nn.Tanh(),
                                        nn.Conv2d(in_channels=4,
                                                   out_channels=4,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                        nn.Tanh(),
                                        nn.Conv2d(in_channels=4,
                                                  out_channels=1,
                                                  kernel_size=kernel,
                                                  padding=padding))
        
        encoder_net_latent = nn.Sequential(nn.Linear(128,256),
                                            nn.ReLU(),
                                            nn.Linear(256,256),
                                            nn.ReLU(),
                                            nn.Linear(256,256),
                                            nn.ReLU(),
                                            nn.Linear(256,128))
        
        self.encoder_net = nn.Sequential(NeuralODEWrapper(encoder_net_data,
                                         **ode_solver_args),
                                         nn.Flatten(start_dim=1, end_dim=-1),
                                         Projection1D(1568, 128),
                                         NeuralODEWrapper(encoder_net_latent, 
                                         **ode_solver_args))
                
        decoder_net_data = nn.Sequential(nn.Conv2d(in_channels=1,
                                                   out_channels=4,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=4,
                                                   out_channels=4,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=4,
                                                   out_channels=1,
                                                   kernel_size=kernel,
                                                   padding=padding))
        decoder_net_latent = nn.Sequential(nn.Linear(128, 256),
                                         nn.Tanh(),
                                         nn.Linear(256, 256),
                                         nn.Tanh(),
                                         nn.Linear(256, 256),
                                         nn.Tanh(),
                                         nn.Linear(256, 128))
        
        self.decoder_net = nn.Sequential(NeuralODEWrapper(decoder_net_latent,
                                         **ode_solver_args),
                                         Projection1D(128, 1568),
                                         nn.Unflatten(-1, (2,28,28)),
                                         NeuralODEWrapper(decoder_net_data, 
                                         **ode_solver_args))
         
    def encoder(self, x):
        return self.encoder_net(x)
 
    def decoder(self, x):
        return self.decoder_net(x)

class AENODEAugConvModel(nn.Module):
    
    def __init__(self, kernel:int, ode_solver_args=None):
 
        super().__init__()
        
        assert kernel % 2 != 0, 'kernel must be odd valued int'
        padding = int(0.5*(kernel - 1))
        kernel = int(kernel)
        
        if ode_solver_args is None: ode_solver_args = {'solver':'dopri5'}
               
        
        encoder_net_data = nn.Sequential(nn.Conv2d(in_channels=2,
                                                       out_channels=4,
                                                       kernel_size=kernel,
                                                       padding=padding),
                                        nn.Tanh(),
                                        nn.Conv2d(in_channels=4,
                                                   out_channels=4,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                        nn.Tanh(),
                                        nn.Conv2d(in_channels=4,
                                                  out_channels=2,
                                                  kernel_size=kernel,
                                                  padding=padding))
        
        encoder_net_latent = nn.Sequential(nn.Linear(128,256),
                                            nn.ReLU(),
                                            nn.Linear(256,256),
                                            nn.ReLU(),
                                            nn.Linear(256,256),
                                            nn.ReLU(),
                                            nn.Linear(256,128))
        
        self.encoder_net = nn.Sequential(nn.Conv2d(1,2, kernel_size=1), 
                                         NeuralODEWrapper(encoder_net_data,
                                         **ode_solver_args),
                                         nn.Flatten(start_dim=1, end_dim=-1),
                                         Projection1D(1568, 128),
                                         NeuralODEWrapper(encoder_net_latent, 
                                         **ode_solver_args))
                
        decoder_net_data = nn.Sequential(nn.Conv2d(in_channels=2,
                                                   out_channels=4,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=4,
                                                   out_channels=4,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=4,
                                                   out_channels=2,
                                                   kernel_size=kernel,
                                                   padding=padding))
        decoder_net_latent = nn.Sequential(nn.Linear(128, 256),
                                         nn.Tanh(),
                                         nn.Linear(256, 256),
                                         nn.Tanh(),
                                         nn.Linear(256, 256),
                                         nn.Tanh(),
                                         nn.Linear(256, 128))
        
        self.decoder_net = nn.Sequential(NeuralODEWrapper(decoder_net_latent,
                                         **ode_solver_args),
                                         Projection1D(128, 1568),
                                         nn.Unflatten(-1, (2,28,28)),
                                         NeuralODEWrapper(decoder_net_data, 
                                         **ode_solver_args),
                                         nn.Conv2d(2,1,1))
         
    def encoder(self, x):
        return self.encoder_net(x)
 
    def decoder(self, x):
        return self.decoder_net(x)
    
    
class VAENODEAugConvModel(nn.Module):
    
    def __init__(self, kernel:int, ode_solver_args=None):
 
        super().__init__()
        
        assert kernel % 2 != 0, 'kernel must be odd valued int'
        padding = int(0.5*(kernel - 1))
        kernel = int(kernel)
        
        if ode_solver_args is None: ode_solver_args = {'solver':'dopri5'}
               
        
        encoder_net_data = nn.Sequential(nn.Conv2d(in_channels=2,
                                                       out_channels=4,
                                                       kernel_size=kernel,
                                                       padding=padding),
                                        nn.Tanh(),
                                        nn.Conv2d(in_channels=4,
                                                   out_channels=4,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                        nn.Tanh(),
                                        nn.Conv2d(in_channels=4,
                                                  out_channels=2,
                                                  kernel_size=kernel,
                                                  padding=padding))
        
        encoder_net_latent = nn.Sequential(nn.Linear(128,256),
                                            nn.ReLU(),
                                            nn.Linear(256,256),
                                            nn.ReLU(),
                                            nn.Linear(256,256),
                                            nn.ReLU(),
                                            nn.Linear(256,128))
        
        self.encoder_net = nn.Sequential(nn.Conv2d(1,2, kernel_size=1), 
                                         NeuralODEWrapper(encoder_net_data,
                                         **ode_solver_args),
                                         nn.Flatten(start_dim=1, end_dim=-1),
                                         Projection1D(1568, 128),
                                         NeuralODEWrapper(encoder_net_latent, 
                                         **ode_solver_args))
                
        decoder_net_data = nn.Sequential(nn.Conv2d(in_channels=2,
                                                   out_channels=4,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=4,
                                                   out_channels=4,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=4,
                                                   out_channels=2,
                                                   kernel_size=kernel,
                                                   padding=padding))
        decoder_net_latent = nn.Sequential(nn.Linear(128, 256),
                                         nn.Tanh(),
                                         nn.Linear(256, 256),
                                         nn.Tanh(),
                                         nn.Linear(256, 256),
                                         nn.Tanh(),
                                         nn.Linear(256, 128))
        
        self.decoder_net = nn.Sequential(NeuralODEWrapper(decoder_net_latent,
                                         **ode_solver_args),
                                         Projection1D(128, 1568),
                                         nn.Unflatten(-1, (2,28,28)),
                                         NeuralODEWrapper(decoder_net_data, 
                                         **ode_solver_args),
                                         nn.Conv2d(2,1,1))
         
    def encoder(self, x):
        
        mean = self.encoder_net(x)
        cov = torch.ones_like(mean)
        
        return self.encoder_net(x), cov
 
    def decoder(self, x):
        
        mean = self.decoder_net(x)
        cov = torch.ones_like(mean)
        
        return self.decoder_net(x), cov 
    
   
    
class AENODEModel(nn.Module):
    
    def __init__(self, input_dims:int, hidden_dims:int, 
                 latent_dims:int, latent_hidden_dims:int, 
                 ode_solver_args=None):
    
        super().__init__()
        
        if ode_solver_args is None: ode_solver_args = {'solver':'dopri5'}
        
        encoder_net_data = nn.Sequential(nn.Linear(input_dims, hidden_dims),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dims, input_dims))
        encoder_net_latent = nn.Sequential(nn.Linear(latent_dims, latent_hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(latent_hidden_dims, latent_dims))
        
        self.encoder_net = nn.Sequential(NeuralODEWrapper(encoder_net_data,
                                         **ode_solver_args),
                                         Projection1D(input_dims, latent_dims),
                                         NeuralODEWrapper(encoder_net_latent, 
                                         **ode_solver_args))

        decoder_net_data = nn.Sequential(nn.Linear(input_dims, hidden_dims),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dims, input_dims))
        decoder_net_latent = nn.Sequential(nn.Linear(latent_dims, latent_hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(latent_hidden_dims, latent_dims))
        
        self.decoder_net = nn.Sequential(NeuralODEWrapper(decoder_net_latent,
                                         **ode_solver_args),
                                         Projection1D(latent_dims, input_dims),
                                         NeuralODEWrapper(decoder_net_data, 
                                         **ode_solver_args))
            
    def encoder(self, x):
        return self.encoder_net(x)
    
    def decoder(self, x):
        return self.decoder_net(x)
    