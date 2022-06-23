# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:02:07 2022

@author: William
"""

import numpy as np

import torch
import torch.nn as nn

from torchdyn.core import NeuralODE


class Projection1D(nn.Module):
    
    def __init__(self, dims_in, dims_out, trainable=False):
        """
        Projection from lower to higher dimensional space. Inverse is calculated
        using the pseudo inverse. Designed for data of dimensions (-1, D) where
        D is the dims_in
        
        Parameters
        ----------
        dims_in : int
            Dimensionality of input data 
        dims_out : int
            Dimensionality of output data
        trainable : bool, optional
            If True the projection matrix can be learnt via backprop, if False 
            the projection corresponds to padding the input data with zeros.
            The default is False.

        Returns
        -------
        None.
        """        
        
        super().__init__()
        
        assert dims_in != dims_out,\
            'Projection1D: dims_in and dims_out should be different sizes'
            
        if dims_in < dims_out:
            projection_matrix = torch.concat([torch.eye(dims_in),
                                              torch.zeros((dims_in, dims_out - dims_in))],
                                             axis=-1)
        else:
            projection_matrix = torch.concat([torch.eye(dims_out),
                                              torch.zeros((dims_out, dims_in - dims_out))],
                                             axis=-1).T
        
        if trainable: self.projection = nn.parameter.Parameter(projection_matrix)
        else:         self.projection = nn.parameter.Parameter(projection_matrix, requires_grad=False)   
        
        self.trainable = trainable
        
    def forward(self, x):
        return x @ self.projection
    
    def inverse(self, x):
        
        if self.trainable: pseudo_inverse = torch.linalg.pinv(self.projection)
        else:              pseudo_inverse = self.projection.T
            
        return x @ pseudo_inverse
    
    
class GradientNetwork(nn.Module):
    
    def __init__(self, dims, hidden_dim=32):
        """
        Simple Network that defines the vector field of the ODE

        Parameters
        ----------
        dims : int 
            Input dimensionality of the data
        hidden_dim : int, optional
            Dimensionality of the hidden layers of the network. The default is 32.

        Returns
        -------
        None.

        """
        
        super().__init__()
        
        self.network = nn.Sequential(
                    nn.Linear(dims, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Linear(hidden_dim, dims))
        
    def forward(self, x):
        return self.network(x)        
    

class CNFAutoEncoderSCurve(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        ode_solver_args = {'sensitivity':'adjoint',
                           'solver':'dopri5',
                           'atol':1e-4,
                           'rtol':1e-4}
        
        #encoder layers
        grad_net_3_encode = GradientNetwork(3, 16)
        grad_net_1_encode = GradientNetwork(1, 8)
        
        self.node_3_encode = NeuralODE(grad_net_3_encode, **ode_solver_args)
        self.projection_encode = Projection1D(3,1)
        self.node_1_encode = NeuralODE(grad_net_1_encode, **ode_solver_args)
        
        #decoder layers
        grad_net_3_decode = GradientNetwork(3, 16)
        grad_net_1_decode = GradientNetwork(1, 8)
        
        self.node_3_decode = NeuralODE(grad_net_3_decode, **ode_solver_args)
        self.projection_decode = Projection1D(1, 3)
        self.node_1_decode = NeuralODE(grad_net_1_decode, **ode_solver_args)


    def encode(self, x):
        
        _, x = self.node_3_encode(x)
        x    = self.projection_encode(x[-1])
        _, x = self.node_1_encode(x)
        
        return x[-1]

    def decode(self, x):
        _, x = self.node_1_decode(x)
        x    = self.projection_decode(x[-1])
        _, x = self.node_3_decode(x)

        return x[-1]

if __name__ == '__main__':
    
    auto = CNFAutoEncoderSCurve()


        
            
