# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:23:25 2022

@author: William Bankes
"""

import torch
import torch.nn as nn
import torch.distributions as dist


class Transform(nn.Module):
    """
    Define a parent class for Transforms to inheret. Will enforce implementation
    of fundamental behaviour.
    
    This should be a layer not an end transform...
    """
    def __init__(self):
        super(Transform, self).__init__()
    
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
    
    def __init__(self, dims, cnst=0.01):
        
        super().__init__()
        
        #define the various weight parameters:
        self.w = nn.parameter.Parameter(torch.ones([dims]).unsqueeze(0))
        self.b = nn.parameter.Parameter(torch.tensor(1.))
        self.v = nn.parameter.Parameter(torch.ones([dims]).unsqueeze(0))
        
        
        
        
        
        

    

if __name__ == '__main__':
    #Create tests for new layers
    pass

        
