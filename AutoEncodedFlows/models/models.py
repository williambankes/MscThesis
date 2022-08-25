# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:02:07 2022

@author: William
"""
import torch
import torch.nn as nn
from torchdyn.nn import DepthCat

from AutoEncodedFlows.models.modules import NeuralODEWrapper, Projection1D
from AutoEncodedFlows.models.modules import SequentialFlow

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
                        #PrintLayer(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.01),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.01),
                        nn.Linear(hidden_dim, dims))

        
        
    def forward(self, x):
        return self.network(x)        
    
class AENODEModel(nn.Module):
    
    def __init__(self, input_dims:int, hidden_dims:int, 
                 latent_dims:int, latent_hidden_dims:int, 
                 ode_solver_args=None):
    
        super().__init__()
        
        if ode_solver_args is None: ode_solver_args = {'solver':'dopri5'}
        
        encoder_net_data = nn.Sequential(nn.Linear(input_dims, hidden_dims),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dims, input_dims))
        encoder_net_latent = nn.Sequential(nn.Linear(latent_dims, latent_hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(latent_hidden_dims, latent_dims))
        
        self.encoder_net = nn.Sequential(NeuralODEWrapper(encoder_net_data,
                                         **ode_solver_args),
                                         Projection1D(input_dims, latent_dims),
                                         NeuralODEWrapper(encoder_net_latent, 
                                         **ode_solver_args))

        decoder_net_data = nn.Sequential(nn.Linear(input_dims, hidden_dims),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dims, hidden_dims),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dims, input_dims))
        decoder_net_latent = nn.Sequential(nn.Linear(latent_dims, latent_hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(latent_hidden_dims, latent_hidden_dims),
                                        nn.Tanh(),
                                        nn.Linear(latent_hidden_dims, latent_dims))
        
        self.decoder_net = nn.Sequential(NeuralODEWrapper(decoder_net_latent,
                                         **ode_solver_args),
                                         Projection1D(latent_dims, input_dims),
                                         NeuralODEWrapper(decoder_net_data, 
                                         **ode_solver_args))
            
    def encoder(self, x):
        return self.encoder_net(x)
    
    def decoder(self, x):
        return self.decoder_net(x)
        
    

class CNFAutoEncoderSCurve(nn.Module):
    
    def __init__(self, trainable=False, orthogonal=False, time_grad=False,
                 input_dims=3, latent_dims=2, 
                 hidden_dim_state=16, hidden_dim_latent=8,
                 t_span=torch.tensor([0.,1.])):
        super().__init__()
        
        #change args to speed up network?
        ode_solver_args = {'sensitivity':'autograd',
                           'solver':'dopri5',
                           'atol':1e-4,
                           'rtol':1e-4}
        
        #encoder layers
        grad_net_3_encode = GradientNetwork(dims=input_dims,
                                            hidden_dim=hidden_dim_state,
                                            time_grad=time_grad)
        grad_net_2_encode = GradientNetwork(dims=latent_dims,
                                            hidden_dim=hidden_dim_latent,
                                            time_grad=time_grad)
                
        self.encode_flow = nn.Sequential(
                            NeuralODEWrapper(grad_net_3_encode, t_span=t_span,
                                             **ode_solver_args),
                            Projection1D(input_dims, latent_dims, trainable=trainable, 
                                             orthog=orthogonal),
                            NeuralODEWrapper(grad_net_2_encode, t_span=t_span,
                                             **ode_solver_args))
        
        #decoder layers
        grad_net_3_decode = GradientNetwork(input_dims,
                                            hidden_dim=hidden_dim_state,
                                            time_grad=time_grad)
        grad_net_2_decode = GradientNetwork(latent_dims, 
                                            hidden_dim=hidden_dim_state,
                                            time_grad=time_grad)
        
        self.decode_flow = nn.Sequential(
                            NeuralODEWrapper(grad_net_2_decode, t_span=t_span,
                                             **ode_solver_args),
                            Projection1D(latent_dims, input_dims, trainable=trainable, 
                                             orthog=orthogonal),
                            NeuralODEWrapper(grad_net_3_decode, t_span=t_span,
                                             **ode_solver_args))
        
    def encode(self, x):
                
        return self.encode_flow(x)

    def decode(self, x):
        
        return self.decode_flow(x)
    
    
class CNFAutoEncoderSCurveAug(nn.Module):
    
    def __init__(self, trainable=False, orthogonal=False, time_grad=False,
                 input_dims=3, aug_dims=4, latent_dims=2, 
                 hidden_dim_state=16, hidden_dim_latent=8,
                 t_span=torch.tensor([0.,1.])):
        super().__init__()
        
        #change args to speed up network?
        ode_solver_args = {#'sensitivity':'adjoint',
                           'sensitivity':'autograd',
                           'solver':'dopri5',
                           'atol':1e-4,
                           'rtol':1e-4}
        
        #encoder layers
        grad_net_state_encode = GradientNetwork(dims=aug_dims,
                                            hidden_dim=hidden_dim_state,
                                            time_grad=time_grad)
        grad_net_latent_encode = GradientNetwork(dims=latent_dims,
                                            hidden_dim=hidden_dim_latent,
                                            time_grad=time_grad)
                
        self.encode_flow = nn.Sequential(
                            Projection1D(input_dims, aug_dims),
                            NeuralODEWrapper(grad_net_state_encode, t_span=t_span,
                                             **ode_solver_args),
                            Projection1D(aug_dims, latent_dims, trainable=trainable, 
                                             orthog=orthogonal),
                            NeuralODEWrapper(grad_net_latent_encode, t_span=t_span,
                                             **ode_solver_args))
        
        #decoder layers
        grad_net_state_decode = GradientNetwork(dims=aug_dims,
                                            hidden_dim=hidden_dim_state,
                                            time_grad=time_grad)
        grad_net_latent_decode = GradientNetwork(latent_dims, 
                                            hidden_dim=hidden_dim_state,
                                            time_grad=time_grad)
        
        self.decode_flow = nn.Sequential(
                            NeuralODEWrapper(grad_net_latent_decode, t_span=t_span,
                                             **ode_solver_args),
                            Projection1D(latent_dims, aug_dims, trainable=trainable, 
                                             orthog=orthogonal),
                            NeuralODEWrapper(grad_net_state_decode, t_span=t_span,
                                             **ode_solver_args),
                            Projection1D(aug_dims, input_dims))
        
    def encode(self, x):
                
        return self.encode_flow(x)

    def decode(self, x):
        
        return self.decode_flow(x)
    
    
class CNFAutoEncoderFlowSCurve(nn.Module):
    
    def __init__(self, trainable=False, orthogonal=False, time_grad=False,
                 input_dims=3, latent_dims=2, 
                 hidden_dim_state=16, hidden_dim_latent=8,
                 t_span=torch.tensor([0.,1.])):
        
        super().__init__()
        
        ode_solver_args = {'sensitivity':'autograd',
                           'solver':'dopri5',
                           'atol':1e-4,
                           'rtol':1e-4}
        
        #Flow layers:        
        grad_net_3 = GradientNetwork(dims=input_dims,
                                     hidden_dim=hidden_dim_state,
                                     time_grad=time_grad)
        grad_net_2 = GradientNetwork(dims=latent_dims,
                                     hidden_dim=hidden_dim_latent,
                                     time_grad=time_grad)
        
        self.flow = SequentialFlow(
            NeuralODEWrapper(grad_net_3, t_span=t_span, **ode_solver_args),
            Projection1D(input_dims, latent_dims, trainable=trainable, 
                             orthog=orthogonal),
            NeuralODEWrapper(grad_net_2, t_span=t_span, **ode_solver_args))

        
    def encode(self, x):
        return self.flow(x)
    
    def decode(self, x):
        return self.flow.inverse(x)
    
    

if __name__ == '__main__':
    
    auto = CNFAutoEncoderSCurve()


        
            
