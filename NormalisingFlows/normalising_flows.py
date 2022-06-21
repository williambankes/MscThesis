# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 08:28:20 2022

@author: William Bankes


Define U -> X as 'forward' where U is the prior space and X is the data space



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
        
            
    def forward_sample(self, n_samples):
        
        samples = self.base_dist.sample((n_samples,))
        x, log_det_J = self.transform.tforward(samples)
    
        return x.detach().numpy(), self.base_dist.log_prob(samples), log_det_J
    
    def sample_KL(self, data, epochs):
        
        """
        Applies equation (13) from [https://arxiv.org/pdf/1912.02762.pdf]. 
        Uncertain if the name in (13) is useful.
        
        Here we call the reverse method of the transform which defines the map
        from X -> U where X is the data space and U the prior space
        
        """        
               
        optim = torch.optim.Adam(self.transform.parameters(), lr=0.0005) 
        
        losses = list()
        
        for epoch in range(epochs):
    
            #Define the forward KL div:
            u, log_detJ = self.transform.treverse(data)
            
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
        
    
    def density_estimation_reverse(self, x):
        """
        Given a sample from the data space X and a transform T return the 
        normalised density of the point under the transform.        

        Parameters
        ----------
        x : torch.tensor()
            (N,D) dimensioned data point

        Returns
        -------
        numpy.array()
            Array of normalised probabilities of the input x under the transform
            T (self.transform)
        """
        
        u, logdet = self.transform.treverse(x)
        return (self.base_dist.log_prob(u) + logdet).detach().numpy()
    
    
class CompositeFlow(nn.Module):
    
    def __init__(self, dims, transform, num):
        
        super().__init__()
        
        self.dims = dims
        self.num = num
        self.flows = nn.ModuleList()
        
        for i in range(num):
            self.flows.append(transform(dims))

    def _transform(self, z, forward):
        
        logdet = torch.zeros([z.shape[0]])
                
        for i in range(self.num):
        
            if forward:    
                z, logdetT = self.flows[i].tforward(z)
            else:
                z, logdetT = self.flows[i].treverse(z)
            
            logdet += logdetT
            
        return z, logdet


    def tforward(self, z):
        return self._transform(z, forward=True)
        
    
    def treverse(self, z):
        return self._transform(z, forward=False)

    
    
    