# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:48:15 2022

@author: William
"""

import torch
from torch.utils import data

from sklearn import datasets

class SCurveDataset(data.Dataset):
    
    def __init__(self, n_samples, extra_dims=0, noise=0.0):
        """
        Pytorch.utils.data.Dataset wrapper of sklearn S curve dataset.

        Parameters
        ----------
        n_samples : int
            number of samples generated from sklearn
        extra_dims : int, optional
            Add <extra_dims> extra dimensions to the data. The default is 0.
        noise : float/double, optional
            Noise added to generate two moons data. The default is 0.0.

        Returns
        -------
        None.

        """
        
        self.n_samples = n_samples
        self.data, self.labels = datasets.make_s_curve(n_samples, noise=0.0)
        self.data = torch.tensor(self.data).float()
        self.labels = torch.tensor(self.labels).float()
                  
        self.data = self.data + torch.tensor([0., -1., 0.])
        
        #extra dimensionality:
        zeros = torch.zeros((self.data.shape[0], extra_dims))
        self.data = torch.concat([self.data, zeros], axis=-1)
        self.extra_dims=extra_dims
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
    def get_dataset(self, colours=False):
        return self.data, self.labels
    
class TwoMoonDataset(data.Dataset):
    
    def __init__(self, n_samples, extra_dims=0, noise=0.0):
        """
        Pytorch.utils.data.Dataset wrapper of sklearn Two Moons dataset.

        Parameters
        ----------
        n_samples : int
            number of samples generated from sklearn
        extra_dims : int, optional
            Add <extra_dims> extra dimensions to the data. The default is 0.
        noise : float/double, optional
            Noise added to generate two moons data. The default is 0.0.

        Returns
        -------
        None.

        """
        
        self.n_samples = n_samples
        self.data, self.labels = datasets.make_moons(n_samples, noise=noise)
        self.data = torch.tensor(self.data).float()
        self.labels = torch.tensor(self.labels).float()
        
        if extra_dims != 0: raise NotImplementedError()
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_dataset(self):
        return self.data, self.labels
    
    
class Manifold1DDataset(data.Dataset):
    
    def __init__(self, n_samples, noise=0.0, func=False):
        """
        Creates a 1D manifold in a 2D space, a function can be passed to enable
        more complex shapes otherwise a plane is used. A degree of uniform random
        noise can be added via: f(x) = noise * U[-1,1]

        Parameters
        ----------
        n_samples : int
            number of samples
        noise : float/double, optional
            Uniform noise added to manifold: U(-1,1)*noise. The default is 0.0.
        func : <function>, optional
            1D vectorised function that enables generation of more interesting
            manifolds. Should take (N,1) input and return (N,1) output, this
            output is concat'd to the input to create (N,2) dataset.
            The default is False.

        Returns
        -------
        None.

        """
        
        data = torch.arange(-1,1, 2/n_samples).reshape(-1, 1)
   
        if func: y = func(data)
        else:    y = torch.zeros((n_samples, 1))
        
        self.data = torch.concat([y, data], axis=-1)
        self.n_samples = n_samples
        self.noise = noise
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_dataset(self):
        return self.data
    
class Manifold1DDatasetNoise(Manifold1DDataset):
    
    def __init__(self, n_samples, noise=0.0, func=False):
        super().__init__(n_samples, noise, func)
        
    def __getitem__(self, idx):
        
        #Clone the data to avoid python view errors
        data_point = self.data[idx].clone()
 
        if isinstance(idx, slice):
            noise = torch.randn(data_point.shape[0])*self.noise
            data_point[:,0] += noise            
        elif isinstance(idx, int):
            noise = torch.randn(1)*self.noise            
            data_point[0] += noise[0]
        else: raise NotImplementedError()
        return data_point
      

    
if __name__ == '__main__':
    
    from torchvision.datasets import FashionMNIST
    from torchvision import transforms
    
    data = FashionMNIST('../', download=True, train=True,
                        transform=transforms.ToTensor())
    sample, target = data[0]
    
    
    """
    import matplotlib.pyplot as plt
    
    #Check that the noise addition doesn't affect the get_dataset method
    dataset = Manifold1DDatasetNoise(n_samples=100, noise=0.1)
    
    #Disply dataset:
    data = dataset.get_dataset()
    fig, axs = plt.subplots(figsize=(15,5), ncols=4)
    
    axs[0].scatter(data[:,0], data[:,1])
    axs[0].set_ylim([-1.5, 1.5])
    axs[0].set_xlim([-1.5, 1.5])
    
    dataset2 = Manifold1DDatasetNoise(n_samples=100, noise=0.07)
    data3 = dataset2[:100]
    
    axs[1].scatter(data3[:,0], data3[:,1])
    axs[1].set_ylim([-1.5, 1.5])
    axs[1].set_xlim([-1.5, 1.5])
    
    
    #Plot data recieved via batches:
    batch_data = dataset[:100]
    axs[2].scatter(batch_data[:,0], batch_data[:,1])
    axs[2].set_ylim([-1.5, 1.5])
    axs[2].set_xlim([-1.5, 1.5])
    
    
    #Replot get dataset:
    dataset = Manifold1DDatasetNoise(n_samples=100, noise=0.15)
    data2 = dataset[:100]
    
    axs[3].scatter(data2[:,0], data2[:,1])
    axs[3].set_ylim([-1.5, 1.5])
    axs[3].set_xlim([-1.5, 1.5])
    
    fig.savefig('./images/Toy.png')
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    