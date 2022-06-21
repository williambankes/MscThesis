# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 08:28:20 2022

@author: William Bankes
"""

import torch
import torch.nn as nn


class NormalisingFlow():
    
    
    def __init__(self, dims, transform, base_dist, 
                 verbose=True, verbose_every=100):
        
        
        self.transform = transform
        self.dims = dims
        self.base_dist = base_dist
        
        #ensure dimensionality of base_dist is correct:
        
        self.verbose = verbose
        self.verbose_every = verbose_every
        
            
    def reverse_KL(self, density_func, epochs, n_samples=100):
        
        optim = torch.optim.Adam(self.transform.parameters())
        losses = list()
        
        for epoch in range(epochs):
        
            #sample from base distribution:
            samples = self.base_dist.sample((n_samples,))
                
            #Push samples through transform 
            x, log_detJ = self.transform(samples)
            
            log_probu = self.base_dist.log_prob(samples)
            
            #how to deal with + 1 term?
            with torch.no_grad():
                log_probx = (density_func(x) + 1).log()
            
            #Calc KL div of sample probabilities and density probabilities
            kl_div = (log_probu - log_detJ - log_probx).mean()
            
            #Optimise:
            optim.zero_grad()
            kl_div.backward()
            optim.step()
            
            losses.append(kl_div.detach().numpy())
            if epoch % 100 == 0 and self.verbose:
                print('NF reverse KL divergence, iter:{} KL div:{}'.\
                      format(epoch, losses[-1]))
                    
        return losses
    
    def reverse_sample(self, n_samples):
        
        samples = self.base_dist.sample((n_samples,))
        x, log_det_J = self.transform(samples)
    
        return x.detach().numpy(), self.base_dist.log_prob(samples), log_det_J
    
    def forward_KL(self, data, epochs):
        
        """
        forward_KL: Applies equation (13) from [https://arxiv.org/pdf/1912.02762.pdf].
        Takes samples from our unknown distribution and applies.
        
        Here transform equivalent to = f^-1(x)
        
        """        
               
        optim = torch.optim.Adam(self.transform.parameters(), lr=0.0005) 
        
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
            if epoch % self.verbose_every == 0 and self.verbose:
                print('NF forward KL divergence, iter:{} KL div:{}'.\
                      format(epoch, losses[-1]))
            
        return losses
        
    
    def density_estimation_forward(self, x):
        
        z, logdet = self.transform(x)
        return (self.base_dist.log_prob(z) + logdet).detach().numpy()
    
    
class CompositeFlow(nn.Module):
    
    def __init__(self, dims, transform, num):
        
        super().__init__()
        
        self.dims = dims
        self.num = num
        self.flows = nn.ModuleList()
        
        for i in range(num):
            self.flows.append(transform(dims))

    def forward(self, z):
        
        logdet = torch.zeros([z.shape[0]])
        
        for i in range(self.num):
            z, logdetT = self.flows[i](z)
            
            logdet += logdetT
            
        return z, logdet

    
    
    