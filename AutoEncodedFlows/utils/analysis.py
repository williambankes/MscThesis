# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:23:20 2022

@author: William
"""

import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from matplotlib import ticker
import matplotlib.pyplot as plt
import torch

pio.renderers.default='browser'

def plotly_3d_point_cloud(model, dataloader):
    
    #Process data:
    dataset = dataloader.dataset
    data, labels = dataset.get_dataset()
    reconstruction = model.decode(model.encode(data)).detach().numpy()    

    #Create pandas dataset:
    cols = ['col_0', 'col_1', 'col_2']
    df_data = pd.DataFrame(data.detach().numpy(), columns=cols)
    df_recon = pd.DataFrame(reconstruction, columns=cols)

    #Create plotly subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

    #Add figures
    fig.add_trace(
        go.Scatter3d(x=df_data[cols[0]],
                     y=df_data[cols[1]],
                     z=df_data[cols[2]],
                     mode='markers',
                     marker=dict(color=labels,
                                 colorscale='bluered')),
        row=1, col=1)
    
    fig.add_trace(
        go.Scatter3d(x=df_recon[cols[0]], 
                     y=df_recon[cols[1]],
                     z=df_recon[cols[2]],
                     mode='markers',
                     marker=dict(color=labels,
                                 colorscale='bluered')),
        row=1, col=2)

    fig.update_layout(title_text="Dataset and Reconstruction")
    fig.show()
    
       
def plotly_3d_point_cloud_scurveAE(model, dataloader):
    
    #Process data:
    dataset = dataloader.dataset
    data, labels = dataset.get_dataset()
    reconstruction = model.decode(model.encode(data)).detach().numpy()
    encoded = model.model.encode_flow[0](data).detach().numpy() 

    #Create pandas dataset:
    cols     = ['col_0', 'col_1', 'col_2']
    df_data  = pd.DataFrame(data.detach().numpy(), columns=cols)
    df_recon = pd.DataFrame(reconstruction, columns=cols)
    df_enc   = pd.DataFrame(encoded, columns=cols)

    #Create plotly subplots
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'},
                {'type': 'scatter3d'}]])

    #Add figures
    fig.add_trace(
        go.Scatter3d(x=df_data[cols[0]],
                     y=df_data[cols[1]],
                     z=df_data[cols[2]],
                     text='DataSet',
                     mode='markers',
                     marker=dict(color=labels,
                                 colorscale='bluered')),
        row=1, col=1)
    
    fig.add_trace(
        go.Scatter3d(x=df_recon[cols[0]], 
                     y=df_recon[cols[1]],
                     z=df_recon[cols[2]],
                     text='Reconstruction',
                     mode='markers',
                     marker=dict(color=labels,
                                 colorscale='bluered')),
        row=1, col=2)
    
    fig.add_trace(
        go.Scatter3d(x=df_enc[cols[0]], 
                     y=df_enc[cols[1]],
                     z=df_enc[cols[2]],
                     text='Encoder Mapping',
                     mode='markers',
                     marker=dict(color=labels,
                                 colorscale='bluered')),
        row=1, col=3)

    fig.update_layout(title_text="Dataset and Reconstruction")
    fig.show()
    
    
def plotly_latent_space_scurveAE(model, dataloader):
    
    #Process data:
    dataset = dataloader.dataset
    data, labels = dataset.get_dataset()
    encoded = model.model.encode_flow(data)
    decoded_latent = model.model.decode_flow[0](encoded)

    encoded = encoded.detach().numpy()
    decoded_latent = decoded_latent.detach().numpy()
    
    #Create pandas dataset:
    cols     = ['col_0', 'col_1']
    df_enc   = pd.DataFrame(encoded, columns=cols)
    df_dec   = pd.DataFrame(decoded_latent, columns=cols)

    #Create plotly subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]])

    #Add figures
    fig.add_trace(
        go.Scatter(x=df_enc[cols[0]],
                     y=df_enc[cols[1]],
                     mode='markers',
                     text='Latent Space',
                     marker=dict(color=labels,
                                 colorscale='bluered')),
        row=1, col=1)
    
    fig.add_trace(
        go.Scatter(x=df_dec[cols[0]], 
                     y=df_dec[cols[1]],
                     mode='markers',
                     text='Latent Decoded Space',
                     marker=dict(color=labels,
                                 colorscale='bluered')),
        row=1, col=2)
    

    fig.update_layout(title_text="Latent Space Visualisation")
    fig.show()
    
    
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


def plotly_3d(points, point_colour, title):
    
    #process data into pandas:
    df = pd.DataFrame(points, columns=['col_0', 'col_1', 'col_2'])
    
    #Deal with colour format:
    
    #Plot 3d scatter plot:
    fig = px.scatter_3d(df, x='col_0', y='col_1', z='col_2')
    fig.show()
    










    
    