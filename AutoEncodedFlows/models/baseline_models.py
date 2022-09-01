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
    
class AEConvModel(nn.Module):
    
    def __init__(self, kernel:int):
        
        super().__init__()
        
        
        assert kernel % 2 != 0, 'kernel must be odd valued int'
        padding = int(0.5*(kernel - 1))
        kernel = int(kernel)
        
        #define encoder and decoder networks:            
        self.encoder_net = nn.Sequential(nn.Conv2d(in_channels=1,
                                                   out_channels=3,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=3,
                                                   out_channels=3,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=3,
                                                   out_channels=1,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                         nn.Flatten(start_dim=1, end_dim=-1),
                                         Projection1D(784, 10),
                                         nn.Linear(10,64),
                                         nn.ReLU(),
                                         nn.Linear(64,64),
                                         nn.ReLU(),
                                         nn.Linear(64,64),
                                         nn.ReLU(),
                                         nn.Linear(64,10))
        
        
        self.decoder_net = nn.Sequential(nn.Linear(10, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 10),
                                         Projection1D(10,784),
                                         nn.Unflatten(-1, (1,28,28)),
                                         nn.Conv2d(in_channels=1,
                                                    out_channels=3,
                                                    kernel_size=kernel,
                                                    padding=padding),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=3,
                                                    out_channels=3,
                                                    kernel_size=kernel,
                                                    padding=padding),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=3,
                                                    out_channels=1,
                                                    kernel_size=kernel,
                                                    padding=padding))
        
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
    
class VAEConvModel(nn.Module):
    
    def __init__(self, kernel:int):
        
        super().__init__()
        
        
        assert kernel % 2 != 0, 'kernel must be odd valued int'
        padding = int(0.5*(kernel - 1))
        kernel = int(kernel)
        
        self.encoder_net = nn.Sequential(nn.Conv2d(in_channels=1,
                                                   out_channels=3,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=3,
                                                   out_channels=3,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=3,
                                                   out_channels=1,
                                                   kernel_size=kernel,
                                                   padding=padding),
                                         nn.Flatten(start_dim=1, end_dim=-1),
                                         Projection1D(784, 10),
                                         nn.Linear(10,64),
                                         nn.ReLU(),
                                         nn.Linear(64,64),
                                         nn.ReLU(),
                                         nn.Linear(64,64),
                                         nn.ReLU())
        
        self.decoder_net = nn.Sequential(nn.Linear(10, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 10),
                                         Projection1D(10,784),
                                         nn.Unflatten(-1, (1,28,28)),
                                         nn.Conv2d(in_channels=1,
                                                    out_channels=3,
                                                    kernel_size=kernel,
                                                    padding=padding),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=3,
                                                    out_channels=3,
                                                    kernel_size=kernel,
                                                    padding=padding),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=3,
                                                    out_channels=1,
                                                    kernel_size=kernel,
                                                    padding=padding))
        
        self.encoder_mean = nn.Linear(64, 10)
        self.encoder_cov = nn.Linear(64, 10)
        
        self.decoder_mean = nn.Conv2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=kernel,
                                       padding=padding)
        self.decoder_cov = nn.Conv2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=kernel,
                                       padding=padding)
        
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

class VAEStdConvModel(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.encoder_net = nn.Sequential(nn.Conv2d(1,4,5,
                                                   padding=0,
                                                   stride=2),
                                         nn.ReLU(),
                                         nn.Conv2d(4,16,5,
                                                   padding=0,
                                                   stride=1),
                                         nn.ReLU(),
                                         nn.Conv2d(16,32,3,
                                                   padding=0,
                                                   stride=1),
                                         nn.ReLU(),
                                         nn.Flatten(start_dim=1, end_dim=-1),
                                         nn.Linear(1152, 10))
        
        self.decoder_net = nn.Sequential(nn.Linear(10,1152),
                                         nn.ReLU(),
                                         nn.Unflatten(-1, (32,6,6)),
                                         nn.ConvTranspose2d(32, 16, 3),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(16, 4, 5),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(4, 1, 6,
                                                            stride=2))
               
        self.encoder_mean = nn.Linear(10,10)
        self.encoder_cov  = nn.Linear(10,10)
        
        self.decoder_mean = nn.Conv2d(1, 1, 5, padding=2)
        self.decoder_cov = nn.Conv2d(1, 1, 5, padding=2)
        
    def encoder(self, x):
        
        x = self.encoder_net(x)
        mean = self.encoder_mean(x)
        cov = self.encoder_cov(x).exp()
        
        return mean, cov
    
    def decoder(self, x):
        
        x = self.decoder_net(x)
        mean = self.decoder_mean(x)
        cov = self.decoder_cov(x).exp()
        
        return mean, cov
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    