# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 08:28:20 2022

@author: William Bankes
"""

import torch


class NormalisingFlow():
    
    def __init__(self, dims, transform, base_dist):
        
        self.transform = transform
        self.dims = dims
        
        self.base_dist = base_dist
        
        #ensure dimensionality of base_dist is correct:
                
    
    def forward_KL(self, data, epochs):
               
        optim = torch.optim.Adam(self.transform.parameters()) 
        
        losses = list()
        
        for epoch in range(epochs):
    
            #Define the forward KL div:
            u, log_detJ = self.transform(data)
            
            #calc log_prob_u under base dist:
            log_probu = self.base_dist.log_prob(u)            
            kl_div = -1 * (log_probu + log_detJ).mean()
            
            #Optimise:
            optim.zero_grad()
            kl_div.backward()
            optim.step()
            
            losses.append(kl_div.detach().numpy())
            if epoch % 10 == 0:
                print('NF forward KL divergence, iter:{} KL div:{}'.\
                      format(epoch, losses[-1]))
            
        return losses
    
    def density_estimation_forward(self, x):
        
        z, logdet = self.transform(x)
        return (self.base_dist.log_prob(z).unsqueeze(-1) + logdet).detach().numpy()