# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:06:27 2022

@author: William
"""

import torch
import wandb
import pytorch_lightning as pl
import torch.nn as nn

from torch.distributions import Normal

class AELearner(pl.LightningModule):
    
    def __init__(self, model:nn.Module, target=False):
        """
        Trainer module for Autoencoder models. Gradient updates defined by the
        reconstruction loss between the encoder and decoder models

        Parameters
        ----------
        model : nn.Module
            AutoEncoder model with encode and decode methods

        Returns
        -------
        None.

        """
        
        super().__init__()
        self.__name__ = 'AELearner'
        self.model = model
        self.target=target
        wandb.watch(model)
        
    def encode(self, x):
        return self.model.encoder(x)
    
    def decode(self, x):
        return self.model.decoder(x)
        
    def forward(self, x):
        return self.model.decoder(self.model.encoder(x))

    def training_step(self, batch, batch_idx):

        if self.target: X = batch[0]            
        else:           X = batch
                       
        encoded = self.model.encoder(X)
        decoded = self.model.decoder(encoded)
        
        #reconstruction loss:
        loss = nn.MSELoss()(decoded, X)
        
        #Log and return metrics:
        wandb.log({'training loss': loss.detach().item()})
        wandb.log({'epoch': self.current_epoch})
        
        return {'loss': loss}
    
    
    def test_step(self, batch, batch_idx):
        
        if self.target: X = batch[0]            
        else:           X = batch
                
        encoded = self.model.encoder(X)
        decoded = self.model.decoder(encoded)
        
        loss = nn.MSELoss()(decoded, X)
        
        wandb.log({'test loss':loss.detach().item()})
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    
class VAELearner(pl.LightningModule):
    
    def __init__(self, model:nn.Module, latent_dims, input_dims, target):
        """
        Pytorch Learner for Variational AutoEncoder model. Encoder and Decoder
        architectures should return torch.dist objects. Dimensions of the output
        should be flattened.
        
        Embed the distributional aspect within the learner to simplify network

        Parameters
        ----------
        encoder : nn.Module
            DESCRIPTION.
        decoder : nn.Module
            DESCRIPTION.
        latent_dims : int
            DESCRIPTION.

        Returns
        -------
        None.
        """
        super().__init__()
        
        self.__name__ = 'VAELearner' 
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.target=target
        
        #Create Jacobian noise dist and base dist:
        mean = torch.zeros(latent_dims).to(self.dev)
        cov = torch.ones(latent_dims).to(self.dev)
        self.prior = Normal(mean, cov)
        
        #Parse input dims:
        if isinstance(input_dims, list):
            self.input_dims = input_dims
            self.multi_dims = True
        elif isinstance(input_dims, int):
            self.input_dims = input_dims
            self.multi_dims = False
        else:
            #Input type not supported:
            raise TypeError()

        self.model = model
        wandb.watch(model)
        
    def encode(self, x):
        return self.model.encoder(x)[0]
    
    def decode(self, x):
        return self.model.decoder(x)[0]
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
        
    def calc_variational_bound(self, batch, batch_idx):
        
        if self.target: X = batch[0]            
        else:           X = batch
        
        #Encode, sample and Deocde sample:
        qzx_mean, qzx_cov = self.model.encoder(X)
        qzx_dist = Normal(qzx_mean, qzx_cov)
        
        z = qzx_dist.rsample()
        
        pxz_mean, pxz_cov = self.model.decoder(z)
        pxz_dist = Normal(pxz_mean, pxz_cov)
        
        if self.multi_dims:
            
            log_qzx = torch.sum(qzx_dist.log_prob(z), axis=1)
            log_pz  = torch.sum(self.prior.log_prob(z), axis=1)
            log_pxz = torch.sum(pxz_dist.log_prob(X), axis=[1,2,3])
                                               
        else:
            
            log_qzx = torch.sum(qzx_dist.log_prob(z), axis=1)
            log_pz = torch.sum(self.prior.log_prob(z), axis=1)
            log_pxz = torch.sum(pxz_dist.log_prob(X), axis=1)

        loss = log_pz + log_pxz - log_qzx
            
        return -loss.mean()
        
    def training_step(self, batch, batch_idx):
        
        loss = self.calc_variational_bound(batch, batch_idx)
    
        wandb.log({'training loss': loss.detach().item()})
        wandb.log({'epoch': self.current_epoch})
        self.log("training_loss", loss)
    
        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        
        if self.target: X = batch[0]            
        else:           X = batch
        
        encoded = self.model.encoder(X)[0]
        decoded = self.model.decoder(encoded)[0]

        loss = nn.MSELoss()(decoded, X)
        
        wandb.log({'test loss':loss.detach().item()})
        self.log("test_loss", loss)
               
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    
    
    
    
