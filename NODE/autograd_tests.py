# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:35:43 2022

Aim to understand how gradients are propogated within the autograd function
of pytorch...

useful links:
    
https://github.com/msurtsukov/neural-ode/blob/master/Neural%20ODEs.ipynb
https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/adjoint.py

autograd pytorch page

https://pytorch.org/docs/stable/generated/torch.autograd.grad.html

extending pytorch page

https://pytorch.org/docs/stable/notes/extending.html

@author: William
"""
#%% imports 

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

#%% Define simple network:

class testNetwork(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.layer = nn.Linear(1, 1)
        
    def forward(self, x):
        
        return self.layer(x)
   
#%% Test simple network:

data = torch.rand((10,1)) 

net = testNetwork()
out = net(data)

#%% Define autograd class: -> do we need to define a new network?
    
class testFunc(Function):
    
    @staticmethod
    def forward(ctx, x, params, net):
        
        ctx.network = net

        with torch.no_grad():
            output = net(x)
            
        ctx.save_for_backward(x, params, output)
            
        return output
    
    @staticmethod
    @once_differentiable    
    def backward(ctx, grad):
        
        net = ctx.network
        x, params, output = ctx.saved_tensors
        
        with torch.enable_grad():
            
            grads = torch.autograd.grad(output,
                                        (x) + params,
                                        allow_unused=True,
                                        retain_graph=True)
        
        return grads
    
#%%


    
#%% Test the autograd class:

params = torch.nn.utils.parameters_to_vector(net.parameters())

#Can we do this via another method? -> wrap in nn.Module to remove .apply method
output = testFunc.apply(data, params, net)



#%% Train the new function using optimizer:
#Try to learn identity mapping    

opt = torch.optim.Adam(net.parameters())
    
for epoch in range(2):
        
    output = testFunc.apply(data, params, net)
    loss = torch.nn.functional.mse_loss(output, data)
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    
    
    
    
        
        
            
    