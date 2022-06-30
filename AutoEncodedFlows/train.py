# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:09:31 2022

Define Pytorch Lightning methods to train the model

@author: William
"""


import pytorch_lightning as pl

import wandb

import torch
import torch.nn as nn
import torch.utils.data as data


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
        
        #clever logging setup in the training step
        
        #Log and return metrics:
        wandb.log({'training loss': loss.detach().item()})
        wandb.log({'epoch': self.current_epoch})
        
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    

if __name__ == '__main__':
    
    from AutoEncodedFlows.models import CNFAutoEncoderSCurve, CNFAutoEncoderFlowSCurve
    from AutoEncodedFlows.datasets import SCurveDataset
    from AutoEncodedFlows.utils import Experiment, wandb_3d_point_cloud, wandb_3d_point_cloud_scurveAE
    
    torch.set_num_threads(14)
    
    trainer_args = {'gpus':1,
                    'max_epochs':20,
                    'enable_checkpointing':False}
    model_args = {}
    dataset_args = {'n_samples':10_000}
    dataloader_args = {'batch_size':508,
                       'shuffle':True}
    
    exp1 = Experiment(project='AutoEncodingFlows',
                      notes='lr=0.001',
                      tags=['MscThesis'],
                      learner=AutoEncoder,
                      model=CNFAutoEncoderSCurve,
                      dataset=SCurveDataset,
                      trainer_args=trainer_args,
                      model_args=model_args,
                      dataset_args=dataset_args,
                      dataloader_args=dataloader_args)
    

    exp1.run()
    exp1.analyse([wandb_3d_point_cloud, wandb_3d_point_cloud_scurveAE])
    exp1.finish()
    
    """    
    
    exp2 = Experiment(project='AutoEncodingFlows',
                      notes='Testing Experiment Class',
                      tags=['MscThesis'],
                      learner=AutoEncoder,
                      model=CNFAutoEncoderFlowSCurve,
                      dataset=SCurveDataset,
                      trainer_args=trainer_args,
                      model_args=model_args,
                      dataset_args=dataset_args,
                      dataloader_args=dataloader_args)
    
    exp2.run()
    exp2.analyse([wandb_3d_point_cloud])
    exp2.finish()
    """
