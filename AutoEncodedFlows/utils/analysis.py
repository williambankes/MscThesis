# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:23:20 2022

@author: William
"""

from matplotlib import ticker
import matplotlib.pyplot as plt

import wandb
import torch


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


def wandb_3d_point_cloud_scurveAE(model, dataloader):
    
    dataset = dataloader.dataset
    data, labels = dataset.get_dataset()
    
    #Specific model analysis
    encoded3D = model.model.encode_flow[0](data)
    colour = label_to_colour(labels[:,None])
    
    recon_points = torch.concat([encoded3D, colour], axis=-1).detach().numpy()
    
    return {'encoded3D': wandb.Object3D(recon_points)}   
    
    
def label_to_colour(labels):
    
    red = torch.tensor([255, 0 ,0])
    blue = torch.tensor([0, 0, 255])
    
    #standardise values:
    alpha = (labels - labels.min())*torch.pi/(2 * labels.max())
    return torch.mul(alpha.sin(), red) + torch.mul(1 - alpha.sin(), blue)
    
    

def plot_3d(points, points_color, title):
    """
    Taken from: 
    https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py

    Parameters
    ----------
    points : np.array
        (N,3) dimensional dataset
    points_color : np.array
        (N) or (N,1) array of colour values
    title : str
        graph title

    Returns
    -------
    None.

    """
    
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()
    
    return fig










    
    