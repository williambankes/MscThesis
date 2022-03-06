# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:23:25 2022

@author: William Bankes
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F


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
        
        #Cholesky decomp scale param:
        alpha = torch.eye(dims) + torch.ones([dims, dims])
            
        self.alpha = nn.parameter.Parameter(alpha)       
        
        #translation
        self.beta = nn.parameter.Parameter(torch.ones([dims]).unsqueeze(0))
        
        #dimension parameters:
        self.dims = dims
        self.cnst = cnst
                
    def log_det_jac(self):
            
        raise NotImplementedError
    
    def forward(self, x):
        
        A = torch.mm(self.alpha, self.alpha.T) + self.cnst*torch.eye(self.dims)
                
        return x @ A + self.beta, torch.det(A).log()
    
class PlanarTransform(Transform):
    
    """
    Questions:
        - if tf.tensordot(self.w, self.u, 1) <= -1 -> can we do this like this?
        - Is the re-param the squared norm or norm?
    """
    
    def __init__(self, dims):
        
        super().__init__()
        
        self.w = nn.parameter.Parameter(torch.ones([dims]).unsqueeze(-1))
        self.b = nn.parameter.Parameter(torch.tensor(1.))
        self.v = nn.parameter.Parameter(torch.ones([dims]).unsqueeze(-1))   
    
    def _h(self, x):
        return torch.tanh(x)
    
    def _h_prime(self, x):
        return 1 - torch.tanh(x).pow(2)
    
    def _v_prime(self):
        
        wTv = self.w.T @ self.v
        m = F.softplus(wTv)
        update = (m - wTv)*(self.w/torch.norm(self.w, p=2)) #is it 2 norm or square?
        
        return self.v + update
    
    def _forward_pass(self, x):
        
        linear = x @ self.w + self.b
        update = self._h(linear) * self._v_prime().T
                
        return x + update
    
    def forward(self, x):
        
        return self._forward_pass(x), self.log_det_jac(x)
    
    def log_det_jac(self, x):
        
        linear = x @ self.w + self.b
        jac = 1. + self._h_prime(linear) * (self.v.T @ self.w)
        det_jac = torch.abs(jac)
        return det_jac.log()
                


if __name__ == '__main__':
    #Create tests for new layers
    pass

        
