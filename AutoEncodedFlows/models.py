# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:02:07 2022

@author: William
"""

import torch
import torch.nn as nn

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
        
        self.trainable, self.dims_in, self.dims_out = trainable, dims_in, dims_out
        self._str = "Projection1D(dims_in={}, dims_out={}, trainable={})".\
            format(self.dims_in, self.dims_out, self.trainable)
            
        
    def forward(self, x):
        return x @ self.projection
    
    def inverse(self, x):
        
        if self.trainable: pseudo_inverse = torch.linalg.pinv(self.projection)
        else:              pseudo_inverse = self.projection.T
            
        return x @ pseudo_inverse
        
    def __repr__(self):
        return self._str

    def __str__(self):
        return self._str
    
class NeuralODEWrapper(nn.Module):
    
    def __init__(self, grad_net, **ode_args):
        
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
        
    def forward(self, x):
        return self.node(x, t_span=torch.tensor([0.,1.]))[-1]

    def inverse(self, x):
        return self.node(x, t_span=torch.tensor([1.,0.]))[-1]
    
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
        grad_net_2_encode = GradientNetwork(2, 8)
        
        self.node_3_encode = NeuralODE(grad_net_3_encode, **ode_solver_args)
        self.projection_encode = Projection1D(3,2)
        self.node_2_encode = NeuralODE(grad_net_2_encode, **ode_solver_args)
        
        #decoder layers
        grad_net_3_decode = GradientNetwork(3, 16)
        grad_net_2_decode = GradientNetwork(2, 8)
        
        self.node_3_decode = NeuralODE(grad_net_3_decode, **ode_solver_args)
        self.projection_decode = Projection1D(2, 3)
        self.node_2_decode = NeuralODE(grad_net_2_decode, **ode_solver_args)

    def encode(self, x):
        
        _, x = self.node_3_encode(x)
        x    = self.projection_encode(x[-1])
        _, x = self.node_2_encode(x)
        
        return x[-1]

    def decode(self, x):
        _, x = self.node_2_decode(x)
        x    = self.projection_decode(x[-1])
        _, x = self.node_3_decode(x)

        return x[-1]
    
    
class CNFAutoEncoderFlowSCurve(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        ode_solver_args = {'sensitivity':'adjoint',
                           'solver':'dopri5',
                           'atol':1e-4,
                           'rtol':1e-4}
        
        #encoder layers
        grad_net_3 = GradientNetwork(3, 16)
        grad_net_2 = GradientNetwork(2, 8)
        
        
        self.flow = SequentialFlow(
            NeuralODEWrapper(grad_net_3, **ode_solver_args),
            Projection1D(3,2),
            NeuralODEWrapper(grad_net_2, **ode_solver_args))

        
    def encode(self, x):
        return self.flow(x)
    
    def decode(self, x):
        return self.flow.inverse(x)
    
    

if __name__ == '__main__':
    
    auto = CNFAutoEncoderSCurve()


        
            
