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
import scipy as ss

from NormalisingFlows.utils import plot_density_contours
from NormalisingFlows.transforms import AffineTransform, PlanarTransform, CNFTransform
from NormalisingFlows.transforms import normalisingODEF
from NormalisingFlows.normalising_flows import NormalisingFlow, CompositeFlow

from NODE.node import ode_solve
from NODE.ode_solver import scipySolver

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
        
                
    @unittest.SkipTest
    def test_affine(self):
        
        #Create Composite flow of Affine Transforms
        affine_flow = CompositeFlow(dims=2, transform=AffineTransform,
                                    num=1)
        nf = NormalisingFlow(dims=2, transform=affine_flow,
                             base_dist=self.base_dist, verbose=False)
        loss = nf.forward_KL(self.data, epochs=5000)
        
        #Check loss decreases 
        self.assertTrue(loss[0] > loss[-1])
        self.visualisations(loss, nf, 'test_affine')          


    def test_planar(self):
            
        planar_flow = CompositeFlow(dims=2, transform=PlanarTransform,
                                    num=1)
        nf = NormalisingFlow(dims=2, transform=planar_flow,
                             base_dist=self.base_dist, verbose=True)
        loss = nf.forward_KL(self.data, epochs=3000)
        
        self.assertTrue(loss[0] > loss[-1])
        self.visualisations(loss, nf, 'test_planar')
        

@unittest.skipIf(dev, "Development mode on")
class TwoMoonsTest(unittest.TestCase):
    
    def setUp(self):
        
        base_mean = torch.zeros(2)
        base_sigma = torch.eye(2)
        self.base_dist = dist.MultivariateNormal(loc=base_mean,
                                            covariance_matrix=base_sigma)
        
        moon_data, _ = make_moons(n_samples=N, noise=0.07)
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
                                    num=16)
        nf = NormalisingFlow(dims=2, transform=planar_flow,
                             base_dist=self.base_dist, verbose_every=1000)
        loss = nf.forward_KL(self.data, epochs=10000)
        
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
                
        self.simple = simple_net(t=True)
        
        self.network = nn.Sequential(
                    nn.Linear(3,3),
                    nn.Sigmoid(),
                    nn.Linear(3,2))
        
        #Control output without training:
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, a=2, b=2.01)
                m.bias.data.fill_(0.01)

        self.network.apply(init_weights)
               
        
    def test_trace_jacobian(self):
        
        #Check normalisingODEF:
        grad_func = normalisingODEF(self.simple)
        t = torch.tensor(0.)
        odef_input = torch.cat([self.data,
                                torch.zeros(self.data.shape[0], 1)],
                               axis=-1)                  
        odef_trace = grad_func(t, odef_input[:2])[0,-1]
        
        #Check jacobian with t included in input
        self.data.requires_grad_()
        t = t.expand(self.data.shape[0], 1) 
        xt = torch.concat([self.data, t], dim=-1) 
        
        jac_output = torch.autograd.functional.jacobian(self.simple,
                                                     xt[:2].reshape(2,-1))
        jac_trace = - torch.trace(jac_output[0,:,0,:])
        
        #Check Jacobian without t included in input
        func = simple_net(t=False)
        func.load_state_dict(self.simple.state_dict())
        output = torch.autograd.functional.jacobian(func, self.data[:2])
        output = torch.trace(output[0,:,0,:])
        
        #assert the two divergences are the same: 
        print('\n My implementation trace:', odef_trace)
        print('\n Jacobian trace:', jac_trace)
        print('\n Jacobian trace -> no time:', - output)
        
        self.assertEqual(odef_trace.item(), jac_trace.item())
        self.assertEqual(jac_trace.item(), -output.item())
        
       
    def test_trace_ivp_sol(self):

        #Create and check the cnf output vs an ivp_solver:
        cnf = CNFTransform(self.network)
        cnf_state, cnf_log_prob = cnf(self.data)
        
        cnf_output = torch.concat([cnf_state,
                                   cnf_log_prob.reshape(-1, 1)], axis=-1)
        
        #Check ivp_scipy solver:
        ode_func_base = normalisingODEF(self.network)
        
        init_data = torch.concat([self.data, 
                                  torch.zeros((self.data.shape[0], 1))],
                                 axis=-1).float()
                
        ode_func = lambda t, x: ode_func_base(torch.tensor(t).float(),
                                           torch.tensor(x).float())\
                    .detach().numpy()
                         
        scipy_output = scipySolver.integrate([0,1], init_data, 3, ode_func)
        scipy_input = torch.tensor(scipy_output[-1]['y'][:,0].reshape(-1,3))
        scipy_output = torch.tensor(scipy_output[-1]['y'][:,-1].reshape(-1,3))               
        
        
        #compare init points -> ensure input points are close
        input_assertion = init_data - scipy_input
        self.assertTrue(input_assertion.max() <= 1e-4)
        
        #compare output points:
        print('\n ivp_solver output', scipy_output)
        print('\n naive solver output', cnf_output)
        
        #understand differences:
        output_assertion_1 = scipy_output[:,:-1] - cnf_output[:,:-1]
        output_assertion_2 = scipy_output[:,-1] - cnf_output[:,-1]
        
        self.assertTrue(output_assertion_1.max() < 1e-4)
        self.assertTrue(output_assertion_2.max() < 1e-4)
        

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
    def test_cnf_prob_output(self):
                
        #Create CNFTransform:
        cnf = CNFTransform(self.simple)
        
        #Pass in data:
        XX, YY = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))

        grid_data = torch.FloatTensor(np.stack((XX.ravel(), YY.ravel())).T)
        _, delta_logp = cnf(grid_data)
        
        #Calc  logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
        logpu = self.base_dist.log_prob(grid_data)
        logpx = (logpu - delta_logp).reshape(XX.shape)
    
        fig, axs = plt.subplots(ncols=2, figsize=(10,5))
        axs[0].contourf(XX, YY, logpu.detach().reshape(XX.shape).exp(), cmap='Blues')
        axs[1].contourf(XX, YY, logpx.detach().exp(), cmap='Blues')        
        
        plt.show()
        
        #integrate over normal dist:
        base_dist_func = lambda x,y: self.base_dist.log_prob(
            torch.tensor([x,y])).exp()

        #Check the distribution integrates to 1:            
        output, _ = ss.integrate.dblquad(base_dist_func, 
                                         a=-4, b=4,
                                         gfun=lambda x: -4, 
                                         hfun=lambda x: 4)
        
        print('\n base dist integral', output)
        
        def output_dist_func(x, y):
            
            data_point = torch.tensor([[x,y]])
            logpu = self.base_dist.log_prob(data_point)
            _, delta_logp = cnf(data_point)
            
            return (logpu - delta_logp).exp()
        
        output, _ = ss.integrate.dblquad(output_dist_func, 
                                         a=-4, b=4,
                                         gfun=lambda x: -4, 
                                         hfun=lambda x: 4)

        print('\n output dist integral', output)
        
        
class constant(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.const = nn.parameter.Parameter(torch.ones([1, 2]))

    def forward(self, x):
        
        return 0*x[:,:-1] + self.const.expand(x.shape[0],2)

class simple_net(nn.Module):
    
    def __init__(self, t=True):
        
        super().__init__()
    
        self.t = t
        self.net = nn.Sequential(
            nn.Linear(2,2),
            nn.Tanh(),
            nn.Linear(2,2))
        
    def forward(self, x):
        
        if self.t:
            return self.net(x[:,:-1])
        else:
            return self.net(x)

@unittest.SkipTest
class CNFTestsTraining(unittest.TestCase):

    def setUp(self):
        
        base_mean = torch.zeros(2)
        base_sigma = torch.eye(2)
        self.base_dist = dist.MultivariateNormal(loc=base_mean,
                                            covariance_matrix=base_sigma)
        
        moon_data, _ = make_moons(n_samples=N, noise=0.01)
        self.data = torch.tensor(moon_data, dtype=torch.float)
                
        
        self.simple = simple_net(t=True)
        
        self.network = nn.Sequential(
                        nn.Linear(3,3),
                        nn.Tanh(),
                        nn.Linear(3,2))
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, a=-1, b=1)
                m.bias.data.fill_(0.01)

        self.network.apply(init_weights)
        self.simple.apply(init_weights)
        
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
    def test_gaussian_cnf(self):
        
        #generate data:
        data = self.base_dist.sample((1000,)) + torch.tensor([1.,1.])
        
        cnf = CNFTransform(self.simple)
        nf = NormalisingFlow(dims=2, transform=cnf,
                             base_dist=self.base_dist, verbose=True)
        loss = nf.forward_KL(data, epochs=500)
        
        self.visualisations(loss, nf, 'test_cnf_gaussian')

        print(cnf.node.func.state_dict())
        
        #where could -'ve objective arise?
        u, logdetJ = nf.transform(self.data)
        
        #prob under base dist:
        probu = self.base_dist.log_prob(u)
        
        # Check Jacobian:
        print(probu[:10])
        print(logdetJ[:10])  
        
        fig, axs = plt.subplots()
        axs.hist(probu.detach())
        axs.hist(logdetJ.detach())
        
    @unittest.SkipTest
    def test_two_moons_cnf(self):
        
        cnf = CNFTransform(self.simple)
        nf = NormalisingFlow(dims=2, transform=cnf,
                             base_dist=self.base_dist, verbose=True)
        loss = nf.forward_KL(self.data, epochs=2000)
        
        self.visualisations(loss, nf, 'test_cnf_gaussian')

        print(cnf.node.func.state_dict())
        

    def test_cnf_normalisiation(self):
        
        N_points = 5000
        
        #sample from base distribution:
        base_samples = self.base_dist.sample((N_points,))
    
        #Create flow and set parameters:        
        cnf = CNFTransform(self.network)
        
        #Create flow and pass through single planar transform:
        nf = NormalisingFlow(dims=2, transform=cnf, base_dist=self.base_dist)
        sample, sample_prob, detJ = nf.reverse_sample(N_points)
                
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
            
            sample_prob_new = sample_prob + detJ.detach().reshape(-1)

            axs[2].hexbin(sample[:,0], 
                       sample[:,1],
                       C=sample_prob_new.exp(),
                       cmap='rainbow')
            axs[2].set_title('CNF Transform (prob adjusted)')
     
        

        
if __name__ == '__main__':
    
    unittest.main(verbosity=2)
    
    