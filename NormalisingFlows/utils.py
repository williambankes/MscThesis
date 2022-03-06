# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:23:59 2022

@author: William Bankes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_density_contours(dense_func, title, colour='Blues'):
    
    fig, axs = plt.subplots(figsize=(7,7))
       
    #create grid:    
    XX, YY = np.meshgrid(np.linspace(-5, 5, 101), np.linspace(-5, 5, 100))

    #Get input from density eval
    py = dense_func(torch.FloatTensor(np.stack((XX.ravel(), YY.ravel())).T))

    #Reshape into 2d output
    ZZ = py.reshape(XX.shape)
        
    #Plot density contours:    
    con = axs.contourf(XX, YY, ZZ, cmap=colour, vmin=ZZ.min(), vmax=ZZ.max())    
    #fig.colorbar(con, ax=axs)
    axs.set_title(title)
    
def plot_density_image(dense_func, title, colour='Blues'):
    
    fig, axs = plt.subplots(figsize=(7,7))
       
    #create grid:    
    XX, YY = np.meshgrid(np.linspace(-5, 5, 101), np.linspace(-5, 5, 100))

    #Get input from density eval
    py = dense_func(torch.FloatTensor(np.stack((XX.ravel(), YY.ravel())).T))

    #Reshape into 2d output
    ZZ = py.reshape(XX.shape)
        
    axs.imshow(ZZ)