# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:36:26 2022

@author: William
"""

import pytorch_lightning as pl

import wandb

import torch
import torch.nn as nn


class AutoEncoder(pl.LightningModule):
    
    def __init__(self, model:nn.Module):
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
        self.__name__ = 'AutoEncoder'
        self.model = model
        wandb.watch(self.model)

    def encode(self, x):
        return self.model.encode(x)
    
    def decode(self, x):
        return self.model.decode(x)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
                
        x = batch
                
        encoded = self.model.encode(x)
        decoded = self.model.decode(encoded)
        
        #reconstruction loss:
        loss = nn.MSELoss()(decoded, x)
        
        #Log and return metrics:
        wandb.log({'training loss': loss.detach().item()})
        wandb.log({'epoch': self.current_epoch})
        
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    

if __name__ == '__main__':
    
    from AutoEncodedFlows.models import CNFAutoEncoderSCurve
    from AutoEncodedFlows.datasets import SCurveDataset
    from AutoEncodedFlows.utils.experiments import Experiment
    from AutoEncodedFlows.utils.wandb_analysis import wandb_3d_point_cloud,\
                                                      wandb_3d_point_cloud_scurveAE         
    from AutoEncodedFlows.utils.analysis import plotly_3d_point_cloud_scurveAE,\
                                                plotly_latent_space_scurveAE,\
                                                plotly_3d_point_cloud
    
    torch.set_num_threads(14)
    
    trainer_args = {'gpus':0,
                    'max_epochs':1,
                    'enable_checkpointing':False}
    model_args = {'trainable':True,
                  'orthogonal':False,
                  'time_grad':True,
                  'hidden_dim_state':32,
                  'hidden_dim_latent':16,
                  't_span':torch.tensor([0.,1.])}
    dataset_args = {'n_samples':10_000}
    dataloader_args = {'batch_size':508,
                       'shuffle':True}
        
    exp1 = Experiment(project='AutoEncodingFlows',
                      tags=['MscThesis', 'AutoEncoder'],
                      learner=AutoEncoder,
                      model=CNFAutoEncoderSCurve,
                      dataset=SCurveDataset,
                      trainer_args=trainer_args,
                      model_args=model_args,
                      dataset_args=dataset_args,
                      dataloader_args=dataloader_args)

    try:
        exp1.run()
        exp1.wandb_analyse([wandb_3d_point_cloud])   #, wandb_3d_point_cloud_scurveAE])
        exp1.analyse([plotly_3d_point_cloud, plotly_latent_space_scurveAE])
        exp1.finish()
    finally:
        exp1.finish()