# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:51:03 2022

@author: William
"""

import torch
import torch.nn as nn
from AutoEncodedFlows.modules import MADE

class VectorFieldMasked(nn.Module):
    
    def __init__(self, dims, hidden_dims):
        """
        Taken from the paper: https://proceedings.mlr.press/v139/bilos21a.html
        as a means of explicitly controlling the trace of the jacobian.    

        Parameters
        ----------
        dims : int
            Input Dimension.
        hidden_dims : int
            Dimension of hidden layers.

        Returns
        -------
        None.

        """
        
    
        super().__init__()
        
        #Define point wise network:
        self.h_net = nn.Sequential(nn.Linear(dims, hidden_dims),
                                   nn.Tanh(),
                                   nn.Linear(hidden_dims, hidden_dims),
                                   nn.Tanh(),
                                   nn.Linear(hidden_dims, hidden_dims),
                                   nn.Tanh(),
                                   nn.Linear(hidden_dims, dims))
            
        #Define autoregressive component:
        self.MADE_net_1 = MADE(dims, [hidden_dims]*2, dims, natural_ordering=False)
        self.MADE_net_2 = MADE(dims, [hidden_dims]*2, dims, natural_ordering=True)
                    
            
        #Define scale parameters:
        self.params = nn.parameter.Parameter(torch.randn(1, dims))    
        
    def forward(self, x):
        
        h_net_output = self.h_net(x)
        pointwise_element = - h_net_output + h_net_output.sum(0)
        inner_element = self.MADE_net_1(x) + self.MADE_net_2(x)
        trace_element = self.params * x
        
        return pointwise_element + inner_element + trace_element


if __name__ == '__main__':

    #Move to test folder at some point:
    data = torch.randn(10,2)

    model = VectorFieldMasked(2,8)

    output = model(data)  

    jac = torch.autograd.functional.jacobian(model, data)
    print('jacobian output\n', output)
    print('jacobian pre train\n', jac[1,:,1,:].diag())      
    print('trace element\n', model.params)
        
        
        
        