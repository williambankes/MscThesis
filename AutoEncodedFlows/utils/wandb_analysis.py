# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:18:20 2022

@author: William
"""


import wandb
import torch
import numpy as np
from torchvision.utils import make_grid

def wandb_3d_point_cloud(model, dataloader):
    """
    Creates a coloured Object3D wandb log   

    Parameters
    ----------
    model : Module/LightningModule
        Module with <encode> and <decode> methods
    dataloader : DataLoader
        Dataloader with <get_dataset> method that returns data and a label

    Returns
    -------
    dict
        wandb Object3D logs for the data space and reconstructed space
    """
    
    dataset = dataloader.dataset
    data, labels = dataset.get_dataset()
    
    reconstruction = model.decode(model.encode(data)).detach()    
    colour = label_to_colour(labels[:, None])
    
    data_points = torch.concat([data, colour], axis=-1).detach().numpy()
    recon_points = torch.concat([reconstruction, colour], axis=-1).detach().numpy()
    
    return {'data points': wandb.Object3D(data_points),
            'reconstruction': wandb.Object3D(recon_points)}

def wandb_image_reconstruction(model, dataloader):
    """
    Logs reconstruction of 25 test images sampled from the dataloader by the model

    Parameters
    ----------
    model : nn.Module/pl.LightningModule
        module with <encode> and <decode> methods
    dataloader : DataLoader
        Test set Dataloader 

    Returns
    -------
    dict
        wandb Image logging object

    """
    #Sample 25 data points from dataloader:
    image, label = next(iter(dataloader))
    recon_image = model.decode(model.encode(image[:25])).detach()        
       
    print('recon_image shape', recon_image.shape)
    #Create a grid of the reconstructed images 
    grid_image = make_grid(recon_image, nrow=5).permute(1,2,0)
    
    #Fix typing:
    grid_image = (255 * grid_image.numpy()).astype(np.uint)
    
    return {"Reconstruction": wandb.Image(grid_image)}
    
    
def wandb_3d_point_cloud_scurveAE(model, dataloader):
    
    dataset = dataloader.dataset
    data, labels = dataset.get_dataset()
    
    #Specific model analysis
    encoded3D = model.model.encode_flow[0](data)
    colour = label_to_colour(labels[:,None])
    
    recon_points = torch.concat([encoded3D, colour], axis=-1).detach().numpy()
    
    return {'encoded3D': wandb.Object3D(recon_points)} 

def label_to_colour(labels):
    """
    Maps single dimensional lables to rgb colours

    Parameters
    ----------
    labels : torch.tensor()
        (N,1) array of float/int types

    Returns
    -------
    torch.tensor()
        (N,3) array of rgb colour values between red and blue
    """
    
    red = torch.tensor([255, 0 ,0])
    blue = torch.tensor([0, 0, 255])
    
    #standardise values:
    alpha = (labels - labels.min())*torch.pi/(2 * labels.max())
    return torch.mul(alpha.sin(), red) + torch.mul(1 - alpha.sin(), blue)

def wandb_manifold1D_scatter_plot(model, dataloader):
    
    torch.cuda.empty_cache()
    
    #Get Data:
    dataset = dataloader.dataset
    data = dataset.get_dataset()    
    
    #Move Data and model onto GPU (due to distributions not having device)
    data = data.cuda() if torch.cuda.is_available() else data.cpu()    
    model = model.cuda() if torch.cuda.is_available() else model
    
    #Process trained model:
    _, output = model(data)
    output = output[-1,:,1:].cpu().detach().numpy()
    
    #Create wandb log from data:
    table = wandb.Table(data=output, columns=['x', 'y'])
    return {"test graph" : wandb.plot.scatter(table, 'x', 'y', title="CNF Transform")}
