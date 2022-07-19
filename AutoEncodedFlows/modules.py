# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 13:30:12 2022

@author: William
"""

import torch
import torch.nn as nn
import numpy as np
from torchdyn.core import NeuralODE

class SequentialFlow(nn.Sequential):
    
    def __init__(self, *args):
        """
        Extension of the nn.Sequential class to allow for an inverse of each 
        Module to be called in reverse order

        Parameters
        ----------
        *args : 
            
        Returns
        -------
        None.

        """        

        super().__init__(*args)
        
        #Could try to ensure that .inverse method must exist
        
        
    def inverse(self, input):
        
        for module in self[::-1]:            
            input = module.inverse(input)
        return input
    
    
class Projection1D(nn.Module):
    
    def __init__(self, dims_in, dims_out, orthog=False, trainable=False):
        """
        Projection from lower to higher dimensional space. Inverse is calculated
        using the pseudo inverse. Designed for data of dimensions (-1, D) where
        D is the dims_in.
        
        -> rotation: https://math.stackexchange.com/questions/2369940/parametric-representation-of-orthogonal-matrices
        i.e. parameterise a rotational matrix via matrix exponential of skew symmetric matrix
        
        Parameters
        ----------
        dims_in : int
            Dimensionality of input data 
        dims_out : int
            Dimensionality of output data
        orthog: bool, optional -> not quite the right term...
            If True the 'identity' projection matrix is preceeded by an orthogonal
            projection mapping with determinant 1.        
        trainable : bool, optional
            If True the projection matrix can be learnt via backprop, if False 
            the projection corresponds to torch.eye padded or sliced.
            The default is False.

        Returns
        -------
        None.
        """        
        
        super().__init__()
        
        assert dims_in != dims_out,\
            'Projection1D: dims_in and dims_out should be different sizes'
        assert not trainable or not orthog,\
            'Projection1D: rotation and trainable should not be both True'
            
        #Create Projection matrix:
        if dims_in < dims_out:
            projection_matrix = torch.concat([torch.eye(dims_in),
                                              torch.zeros((dims_in, dims_out - dims_in))],
                                             axis=-1)
        else:
            projection_matrix = torch.concat([torch.eye(dims_out),
                                              torch.zeros((dims_out, dims_in - dims_out))],
                                             axis=-1).T
                
        if orthog: 
            #self.rotations = nn.parameter.Parameter(torch.randn((dims_in - 1)))
            #self.indices   = np.array([(i+1, i) for i in range(dims_in - 1)])
            self.orthog_params = nn.parameter.Parameter(torch.rand(dims_in, dims_in))
            self.diag_params = nn.parameter.Parameter(torch.rand(dims_in))
            self.indices = np.array([(i,i) for i in range(dims_in)])
            
        
        if trainable: self.projection = nn.parameter.Parameter(projection_matrix)
        else:         self.projection = nn.parameter.Parameter(projection_matrix, requires_grad=False)   
        
        self.trainable, self.orthog = trainable, orthog
        self.dims_in, self.dims_out = dims_in, dims_out
        self._str = "Projection1D(dims_in={}, dims_out={}, orthog={}, trainable={})".\
            format(self.dims_in, self.dims_out, self.orthog, self.trainable)
                
    def orthogonal_matrix(self):
        
        """
        tri_matrix = torch.zeros((self.dims_in, self.dims_in), 
                                 device=self.rotations.device)
        tri_matrix[self.indices[:,0], self.indices[:,1]] = self.rotations
        skew_matrix = tri_matrix - tri_matrix.T 
        """
        
        diag_matrix = torch.zeros((self.dims_in, self.dims_in), 
                                  device=self.orthog_params.device)
        diag_matrix[self.indices[:,0], self.indices[:,1]] = self.diag_params
        tri_matrix = self.orthog_params.triu()
        skew_matrix = tri_matrix - tri_matrix.T + self.diag_params
        
        return torch.matrix_exp(skew_matrix)
        
    def forward(self, x):
        
        if self.orthog: return x @ self.orthogonal_matrix() @ self.projection
        else: return x @ self.projection
    
    def inverse(self, x):
                
        if self.trainable: return x @ torch.linalg.pinv(self.projection)
        elif self.orthog:  return x @ torch.linalg.pinv(
                self.orthogonal_matrix() @ self.projection)
        else: return x @ self.projection.T
        
        
    def __repr__(self):
        return self._str

    def __str__(self):
        return self._str
    
    
class NeuralODEWrapper(nn.Module):
    
    def __init__(self, grad_net, t_span=torch.tensor([0., 1.]), **ode_args):
        
        """
        Wrapper around NeuralODE to implement <inverse> method and remove time
        evals from solution

        Parameters
        ----------
        grad_net : nn.Module
            nn.Module that defines the vector field of the NeuralODE
        **ode_args: dict
            Torchdyn ode solver args (see documentation)

        Returns
        -------
        None.
        
        """

        super().__init__()
        self.node = NeuralODE(grad_net, return_t_eval=False, **ode_args)
        self.t_span = t_span
        
    def forward(self, x):
        return self.node(x, t_span=self.t_span)[-1]

    def inverse(self, x):
        return self.node(x, t_span=self.t_span)[-1]
    
class PrintLayer(nn.Module):
    
    def __init__(self, name):
        """
        Debugging layer. Returns the input and prints the name given upon
        instantiation.

        Parameters
        ----------
        name : str
            Debug message to print when called

        Returns
        -------
        None.

        """
        
        super().__init__()
        self.name = name
        
    def forward(self, x):
        print(self.name)
        return x
    
    