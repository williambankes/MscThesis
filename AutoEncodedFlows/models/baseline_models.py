# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:31:09 2022

@author: William
"""

import torch.nn as nn
from AutoEncodedFlows.models.modules import Projection1D

class AELinearModel(nn.Module):
    
    def __init__(self, input_dims:int, hidden_dims:int, 
                 latent_dims:int, latent_hidden_dims:int):
        
        super().__init__()
        
        self.encoder_net = nn.Sequential(nn.Linear(input_dims, hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dims, input_dims),
                                         Projection1D(input_dims, latent_dims),
                                         nn.Linear(latent_dims, latent_hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(latent_hidden_dims, latent_dims))
        
        self.decoder_net = nn.Sequential(nn.Linear(latent_dims, latent_hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(latent_hidden_dims, latent_dims),
                                         Projection1D(latent_dims, input_dims),
                                         nn.Linear(input_dims, hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dims, input_dims))
        
    def encoder(self, x):
        return self.encoder_net(x)
    
    def decoder(self, x):
        return self.decoder_net(x)
        
        
class VAELinearModel(nn.Module):

    def __init__(self, input_dims:int, hidden_dims:int, 
                 latent_dims:int, latent_hidden_dims:int):
        
        super(VAELinearModel, self).__init__()
        
        self.encoder_net = nn.Sequential(nn.Linear(input_dims, hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dims, input_dims),
                                         Projection1D(input_dims, latent_dims),
                                         nn.Linear(latent_dims, latent_hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                         nn.ReLU())
        
        self.decoder_net = nn.Sequential(nn.Linear(latent_dims, latent_hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(latent_hidden_dims, latent_dims),
                                         Projection1D(latent_dims, input_dims),
                                         nn.Linear(input_dims, hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.ReLU())
        
        self.encoder_mean = nn.Linear(latent_hidden_dims, latent_dims)
        self.encoder_cov = nn.Linear(latent_hidden_dims, latent_dims)
        
        self.decoder_mean = nn.Linear(hidden_dims, input_dims)
        self.decoder_cov = nn.Linear(hidden_dims, input_dims)
    
    def encoder(self, x):
        
        net_output = self.encoder_net(x)
        mean = self.encoder_mean(net_output)
        cov = self.encoder_cov(net_output)
        
        return mean, cov.exp()
    
    def decoder(self, x):
        
        net_output = self.decoder_net(x)
        mean = self.decoder_mean(net_output)
        cov = self.decoder_cov(net_output)
        
        return mean, cov.exp()

