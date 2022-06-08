# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:24:50 2022

A set of testing environments used for development and sense checking code, not
unit tests per say but uses the framework to structure and run code without bugs

To do:
    
- Investigate stabilising planar flow transforms with batch layers

- Density issues with CNF -> Sums to strange values


@author: William Bankes
"""

#%% Imports
import unittest
import torch
import torch.nn as nn
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

from NormalisingFlows.utils import plot_density_contours
from NormalisingFlows.transforms import AffineTransform, PlanarTransform, CNFTransform
from NormalisingFlows.normalising_flows import NormalisingFlow, CompositeFlow

from sklearn.datasets import make_moons
  
#%% Global Test variables:
    
N = 1000
visualise = True
dev = True

#%%

@unittest.skipIf(dev, "Development mode on")
class GaussianTests(unittest.TestCase):
    
    """
    Test transforms and Normalising Flow setup in simple setting of Gaussian
    noise
    """
    def setUp(self):
        
        """
        Generates target and base distributions        
        """
        
        #define Gaussian distributions
        target_mean = torch.ones([1,2])
        target_sigma = torch.eye(2) * 4
        self.target_dist = dist.MultivariateNormal(loc=target_mean,
                                              covariance_matrix=target_sigma)
        
        base_mean = torch.zeros(2)
        base_sigma = torch.eye(2)
        self.base_dist = dist.MultivariateNormal(loc=base_mean,
                                            covariance_matrix=base_sigma)
        
        #generate data:
        global N
        data = self.target_dist.sample(torch.Size([N]))
        self.data = data.reshape(N, 2)
        
        
    def visualisations(self, loss, nf, name):
        
        global visualise
        if visualise:
            
            fig, axs = plt.subplots(figsize=(10,7))
            axs.plot(loss)
            axs.set_title('GaussianTests.{} training loss'.format(name))
    
            # Plot actual and 'found' distributions:
            plot_density_contours(lambda x: np.exp(nf.density_estimation_forward(x)),
                                  'GaussianTests.{} backward dist'.format(name))
            plot_density_contours(lambda x: self.target_dist.log_prob(x).exp(),
                                  'GaussianTests.{} target dist'.format(name))
        
                
    
    def test_affine(self):
        
        #Create Composite flow of Affine Transforms
        affine_flow = CompositeFlow(dims=2, transform=AffineTransform,
                                    num=1)
        nf = NormalisingFlow(dims=2, transform=affine_flow,
                             base_dist=self.base_dist, verbose=False)
        loss = nf.forward_KL(self.data, epochs=1000)
        
        #Check loss decreases 
        self.assertTrue(loss[0] > loss[-1])
        self.visualisations(loss, nf, 'test_affine')          

            
    def test_planar(self):
            
        planar_flow = CompositeFlow(dims=2, transform=PlanarTransform,
                                    num=1)
        nf = NormalisingFlow(dims=2, transform=planar_flow,
                             base_dist=self.base_dist, verbose=False)
        loss = nf.forward_KL(self.data, epochs=1000)
        
        self.assertTrue(loss[0] > loss[-1])
        self.visualisations(loss, nf, 'test_planar')
        

@unittest.skipIf(dev, "Development mode on")
class TwoMoonsTest(unittest.TestCase):
    
    def setUp(self):
        
        base_mean = torch.zeros(2)
        base_sigma = torch.eye(2)
        self.base_dist = dist.MultivariateNormal(loc=base_mean,
                                            covariance_matrix=base_sigma)
        
        moon_data, _ = make_moons(n_samples=N, noise=0.01)
        self.data = torch.tensor(moon_data, dtype=torch.float)
               
           
    def visualisations(self, loss, nf, name):
        
        global visualise
        if visualise:
            
            fig, axs = plt.subplots(figsize=(10,7))
            axs.plot(loss)
            axs.set_title('TwoMoonsTests.{} training loss'.format(name))
    
            # Plot actual and 'found' distributions:
            plot_density_contours(lambda x: np.exp(nf.density_estimation_forward(x)),
                                  'TwoMoonsTests.{} backward dist'.format(name))
        
        
    def test_planar_ensemble(self):
        
        planar_flow = CompositeFlow(dims=2, transform=PlanarTransform,
                                    num=10)
        nf = NormalisingFlow(dims=2, transform=planar_flow,
                             base_dist=self.base_dist, verbose=True)
        loss = nf.forward_KL(self.data, epochs=3000)
        
        self.assertTrue(loss[0] > loss[-1])
        
        self.visualisations(loss, nf, 'test_planar_ensemble')
        
    def test_planar_normalisation(self):
        
        #sample from base distribution:
        base_samples = self.base_dist.sample((N,))
    
        #Create flow and set parameters:        
        flow = CompositeFlow(dims=2, transform=PlanarTransform, num=1)
        flow.flows[0].w = nn.parameter.Parameter(torch.tensor([[5.,1.]]).T)
        flow.flows[0].v = flow.flows[0].w
        
        #Create flow and pass through single planar transform:
        nf = NormalisingFlow(dims=2, transform=flow, base_dist=self.base_dist)
        sample, sample_prob, detJ = nf.reverse_sample(N)
        
        global visualise  
        if visualise:
            
            fig, axs = plt.subplots(ncols=3, figsize=(15,5))
            axs[0].hexbin(base_samples[:,0], 
                       base_samples[:,1],
                       C=self.base_dist.log_prob(base_samples).exp(),
                       cmap='rainbow')
            axs[0].set_title('Base Distribution')
            
        
            axs[1].hexbin(sample[:,0],
                       sample[:,1],
                       C=sample_prob.exp(),
                       cmap='rainbow')
            axs[1].set_title('Transformed')
            
            
            sample_prob_new = sample_prob - detJ.detach().reshape(-1)

            axs[2].hexbin(sample[:,0], 
                       sample[:,1],
                       C=sample_prob_new.exp(),
                       cmap='rainbow')
            axs[2].set_title('Normalised Transform')
             

class CNFTests(unittest.TestCase):
    
    """
    Test Continuous Normalising Flow implementation
    """
    
    def setUp(self):
        
        base_mean = torch.zeros(2)
        base_sigma = torch.eye(2)
        self.base_dist = dist.MultivariateNormal(loc=base_mean,
                                            covariance_matrix=base_sigma)
        
        moon_data, _ = make_moons(n_samples=N, noise=0.01)
        self.data = torch.tensor(moon_data, dtype=torch.float)
                
        
        self.network_simple = nn.Sequential(
                    nn.Linear(3,3),
                    nn.Sigmoid(),
                    nn.Linear(3,2))
        
        
        self.network = nn.Sequential(
                    nn.Linear(3,32),
                    nn.BatchNorm1d(32),
                    nn.LeakyReLU(),
                    nn.Linear(32,32),
                    nn.BatchNorm1d(32),
                    nn.LeakyReLU(),
                    nn.Linear(32,2))

        #Ensure smart initialisation of network parameters        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        net.apply(init_weights)

        
        
    def visualisations(self, loss, nf, name):
        
        global visualise
        if visualise:
            
            fig, axs = plt.subplots(figsize=(10,7))
            axs.plot(loss)
            axs.set_title('TwoMoonsTests.{} training loss'.format(name))
    
            # Plot actual and 'found' distributions:
            plot_density_contours(lambda x: np.exp(nf.density_estimation_forward(x)),
                                  'TwoMoonsTests.{} backward dist'.format(name))
  
    @unittest.SkipTest
    def test_cnf_output_dims(self):
                
        #Create cnfTransform:
        cnf = CNFTransform(self.network)
        pf = PlanarTransform(2)
        
        cnf_out, cnf_log_prob = cnf(self.data)
        pf_out, pf_log_prob = pf(self.data)
        
        #Ensure dimensionality of outputs correct
        self.assertTrue(cnf_out.shape == pf_out.shape)
        self.assertTrue(cnf_log_prob.shape == pf_log_prob.shape)
          
    @unittest.SkipTest     
    def test_cnf(self):
        
        cnf = CNFTransform(self.network)
        nf = NormalisingFlow(dims=2, transform=cnf,
                             base_dist=self.base_dist, verbose=True)
        loss = nf.forward_KL(self.data, epochs=1000)
        
        #self.assertTrue(loss[0] > loss[-1])
        self.visualisations(loss, nf, 'test_cnf')
        
    def test_gaussian_cnf(self):
        
        #generate data:
        data = self.base_dist.sample((1000,))
        
        cnf = CNFTransform(self.network)
        nf = NormalisingFlow(dims=2, transform=cnf,
                             base_dist=self.base_dist, verbose=True)
        loss = nf.forward_KL(data, epochs=300)
        
        self.visualisations(loss, nf, 'test_cnf_gaussian')
      
      
      
    @unittest.SkipTest
    def test_cnf_normalisiation(self):
        
        #sample from base distribution:
        base_samples = self.base_dist.sample((5000,))
    
        #Create flow and set parameters:        
        cnf = CNFTransform(self.network)
        
        #Create flow and pass through single planar transform:
        nf = NormalisingFlow(dims=2, transform=cnf, base_dist=self.base_dist)
        sample, sample_prob, detJ = nf.reverse_sample(N)
        
        print(detJ)
        
        global visualise  
        if visualise:
            
            fig, axs = plt.subplots(ncols=3, figsize=(15,5))
            
            axs[0].hexbin(base_samples[:,0], 
                       base_samples[:,1],
                       C=self.base_dist.log_prob(base_samples).exp(),
                       cmap='rainbow')
            axs[0].set_title('Base Distribution')
            
            axs[1].hexbin(sample[:,0],
                       sample[:,1],
                       C=sample_prob.exp(),
                       cmap='rainbow')
            axs[1].set_title('CNF Transformed (prob unadjusted)')
            
            sample_prob_new = sample_prob - detJ.detach().reshape(-1)

            axs[2].hexbin(sample[:,0], 
                       sample[:,1],
                       C=sample_prob_new.exp(),
                       cmap='rainbow')
            axs[2].set_title('CNF Transform (prob adjusted)')
     
            
        
if __name__ == '__main__':
    
    unittest.main(verbosity=2)
  