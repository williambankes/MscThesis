# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:09:31 2022

Define Pytorch Lightning methods to train the model

@author: William
"""


import pytorch_lightning as pl

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
        self.model = model


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
                
        x = batch
        
        encoded = self.model.encode(x)
        decoded = self.model.decode(encoded)
        
        #reconstruction loss:
        loss = nn.MSELoss()(decoded, x)
        
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

if __name__ == '__main__':
    
    from AutoEncodedFlows.models import CNFAutoEncoderSCurve
    from AutoEncodedFlows.datasets import SCurveDataset
    
    #Set cpu core threads
    torch.set_num_threads(14)
    
    s_curve_dataset = SCurveDataset(10_000)
    s_curve_dataloader = data.DataLoader(s_curve_dataset, batch_size=254, shuffle=True)
    
    autoencoder = AutoEncoder(CNFAutoEncoderSCurve())
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(autoencoder, train_dataloaders=s_curve_dataloader)

