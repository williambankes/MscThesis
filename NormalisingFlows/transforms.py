# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:23:25 2022

@author: William Bankes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from NODE.node import ODEF, NeuralODE


class Transform(nn.Module):
    """
    Define a parent class for Transforms to inheret. Will enforce implementation
    of fundamental behaviour.
    
    This should be a layer not an end transform...
    """
    def __init__(self):
        super().__init__()
    
    def log_det_jac(self, x):
        
        raise NotImplementedError
        
    def forward(self):
        
        raise NotImplementedError  
        
    
class AffineTransform(Transform):
    
    def __init__(self, dims, cnst=0.01):
        super().__init__()
        
        #Init Parameters
        alpha = torch.eye(dims) + torch.ones([dims, dims])
        alpha = alpha/4
            
        self.alpha = nn.parameter.Parameter(alpha)       
        self.beta = nn.parameter.Parameter(torch.ones([dims]).unsqueeze(0)/2)
        
        #dimension parameters:
        self.dims = dims
        self.cnst = cnst
                
    def log_det_jac(self):
            
        raise NotImplementedError
    
    def forward(self, x):
        
        A = torch.mm(self.alpha, self.alpha.T) + self.cnst*torch.eye(self.dims)
                
        return x @ A + self.beta, torch.det(A).log()
    
class PlanarTransform(Transform):
        
    def __init__(self, dims):
        
        super().__init__()
        
        self.w = nn.parameter.Parameter(torch.rand([dims, 1]) * 2 - 1)
        self.b = nn.parameter.Parameter(torch.tensor(0.))
        self.v = nn.parameter.Parameter(torch.rand([dims, 1]) * 2 - 1)
        
    def _h(self, x):
        return torch.tanh(x)
    
    def _h_prime(self, x):
        return 1. - torch.tanh(x).pow(2)
    
    def _v_prime(self):
                        
        wTv = self.w.T @ self.v
                
        m = -1. + F.softplus(wTv)
        update = (m - wTv)*(self.w/(self.w.T@self.w))
    
        return self.v + update
    
    def _forward_pass(self, x):
        
        linear = x @ self.w + self.b
        update = self._h(linear) * self._v_prime().T
                        
        return x + update
    
    def forward(self, x):
        
        return self._forward_pass(x), self.log_det_jac(x)
    
    def log_det_jac(self, x):
        
        linear = x @ self.w + self.b
        jac = 1. + self._h_prime(linear) * (self.w.T @ self._v_prime())
        det_jac = torch.abs(jac) + 1e-4
        
        return det_jac.log().reshape(-1)

    
    
class normalisingODEF(ODEF):
    """
    Network parameterises the Neural ODE classifier.
    """
    
    def __init__(self, network):
        super().__init__()
        self.net = network
        
                
    def _calc_trace_dfdx(self, f, x):
        
        """Calculates the trace of the Jacobian df/dz.
        Taken from: torchdiffeq/examples/cnf.py
        
        f:<torch.tensor>
        (N,D) output of ode solver
        
        x:<torch.tensor>
        (N,D) input of ode solver, gradient enabled
        """
               
        sum_diag = 0.
        for i in range(x.shape[1]):
            sum_diag += torch.autograd.grad(f[:, i].sum(),
                                            x,
                                            create_graph=True,
                                            allow_unused=True)[0][:, i]
        return sum_diag.reshape(-1, 1)
        
        
    def forward(self, t, x):
                       
        #Split input into state and logp init
        data = x[:,:-1]
        
        with torch.set_grad_enabled(True):
        
            data.requires_grad_()
            t = t.expand(x.shape[0], 1) 
            xt = torch.concat([data, t], dim=-1)        
            f = self.net(xt)
            
            dlogpdt = - self._calc_trace_dfdx(f, data)
                
        return torch.concat([f, dlogpdt], axis=-1)
    
    
class CNFTransform(Transform):
    
    """
    Neural ODE transform
    """
    
    def __init__(self, network):
        super().__init__()
        self.node = NeuralODE(normalisingODEF(network))
    
        
    def forward(self, x):
                    
        #Add log_prob dimension:
        prob_init = torch.zeros((x.shape[0], 1))
        x = torch.cat([x, prob_init], axis=-1)
        
        x = self.node(x)
        
        output_state = x[:, :-1]
        log_det_J = x[:, -1]
        
        return output_state, log_det_J
        

                     