# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:02:07 2022

@author: William
"""
import torch
import torch.nn as nn
from torchdyn.nn   import DepthCat

from AutoEncodedFlows.modules import NeuralODEWrapper, Projection1D
from AutoEncodedFlows.modules import SequentialFlow


class GradientNetwork(nn.Module):
    
    def __init__(self, dims, time_grad=False, hidden_dim=32):
        """
        Simple Network that defines the vector field of the ODE
        
        -> Add time and more complexity
        
        -> Why doesn't dynamic layers work?

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
        
        """
        #Create init layer:
        layers = self._create_layer(dims, hidden_dim, time_grad)
        
        #Create layers:
        for l in range(n_layers):
            layers.extend(self._create_layer(hidden_dim, hidden_dim, time_grad))
        
        #final layer:
        if time_grad: layers.extend([nn.DepthCat(1), nn.Linear(hidden_dim + 1, dims)])
        else:         layers.append(nn.Linear(hidden_dim, dims))
        
        self.network = nn.Sequential(*layers)        
        
        """
        if time_grad:
            self.network = nn.Sequential(
                        DepthCat(1),
                        nn.Linear(dims + 1, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.01),
                        DepthCat(1),
                        nn.Linear(hidden_dim + 1, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.01),
                        DepthCat(1),
                        nn.Linear(hidden_dim + 1, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.01),
                        DepthCat(1),
                        nn.Linear(hidden_dim + 1, dims))
        else:
            self.network = nn.Sequential(
                        nn.Linear(dims, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.01),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.01),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.01),
                        nn.Linear(hidden_dim, dims))

        
    def _create_layer(self, dims_in, dims_out, time_grad):
        
        if time_grad: layer = [DepthCat(1),
                               nn.Linear(dims_in + 1, dims_out),
                               nn.BatchNorm1d(dims_out),
                               nn.LeakyReLU(0.01)]
        else:         layer = [nn.Linear(dims_in, dims_out),
                               nn.BatchNorm1d(dims_out),
                               nn.LeakyReLU(0.01)]   
        
        return layer
        
    def forward(self, x):
        return self.network(x)        
    

class CNFAutoEncoderSCurve(nn.Module):
    
    def __init__(self, trainable=False, orthogonal=False, time_grad=False,
                 hidden_dim_state=16, hidden_dim_latent=8,
                 t_span=torch.tensor([0.,1.])):
        super().__init__()
        
        #change args to speed up network?
        ode_solver_args = {'sensitivity':'adjoint',
                           'solver':'dopri5',
                           'atol':1e-4,
                           'rtol':1e-4}
        
        #encoder layers
        grad_net_3_encode = GradientNetwork(dims=3, hidden_dim=hidden_dim_state,
                                            time_grad=time_grad)
        grad_net_2_encode = GradientNetwork(dims=2, hidden_dim=hidden_dim_latent,
                                            time_grad=time_grad)
                
        self.encode_flow = nn.Sequential(
                            NeuralODEWrapper(grad_net_3_encode, t_span=t_span,
                                             **ode_solver_args),
                            Projection1D(3,2, trainable=trainable, 
                                             orthog=orthogonal),
                            NeuralODEWrapper(grad_net_2_encode, t_span=t_span,
                                             **ode_solver_args))
        
        #decoder layers
        grad_net_3_decode = GradientNetwork(3, 16)
        grad_net_2_decode = GradientNetwork(2, 8)
        
        self.decode_flow = nn.Sequential(
                            NeuralODEWrapper(grad_net_2_decode, t_span=t_span,
                                             **ode_solver_args),
                            Projection1D(2,3, trainable=trainable, 
                                             orthog=orthogonal),
                            NeuralODEWrapper(grad_net_3_decode, t_span=t_span,
                                             **ode_solver_args))
        
    def encode(self, x):
                
        return self.encode_flow(x)

    def decode(self, x):
        
        return self.decode_flow(x)
    
    
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


        
            
