# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:36:46 2022

TO DO:
- Generalise solver to >2 dim data, see _prep_func

@author: William
"""

import torch
from scipy.integrate import solve_ivp

class scipySolver():
  
    @staticmethod
    def integrate(t_span, x_init, dims, func, t_eval=None, *func_args):
        """
        Numerical Integration of self.func across some span of time and for some
        data x of dimensions [Batch x dims]
        
        t_span:list()
        Some iterable defining the bounds of the integration
        
        x_init:torch.tensor()
        [Batch x dims] tensor containing the initial points of the integration

        func: <method> (t, x) where x is the state

        Returns
        -------
        success: bool
        Success flag
        
        x_out:torch.tensor()
        Ouput tensor of the integration
        
        t:torch.tensor()
        Output tensor of the time points evaluated by the integrator
        """
        
        #assertion that x is correctly dimensioned:
        assert len(x_init.shape) == 2,\
            'Tensors with more than two dimensions not implemented'
        assert x_init.shape[1] == dims,\
            "Input tensors dims: {} don't match construct dims: {}".format(x_init.shape[1],
                                                                           dims)
        
        func = scipySolver._prep_func(func, dims)
            
            
        #record the batch size
        batch = x_init.shape[0]
            
        #Run the solver_ivp function:
        solution = solve_ivp(func,
                              t_span,
                              x_init.reshape(-1),
                              args=func_args,
                              t_eval=t_eval)
        
        #re map the 1-d solution back to input dims:
        x_out = solution['y'].reshape(-1, batch, dims)
        t = solution['t']
        success = solution['success']
        
        return success, x_out, t, solution

    @staticmethod
    def _prep_func(func, dims):
        """
        We start with a (N,D) func input, we need to move to (NxD) func for the 
        solver
        
        Parameters
        ----------
        func: <method>
        Function with some callable(t, x, *args) style input

        dims: int
        Size of the non-batch dimension input x
            
        Returns
        -------
        new_func: <method>
        Implementation of func that takes 1-dim inputs and returns 1-dim outputs
        """
        
        def new_func(t, x, *args):
            
            #take 1 dim output and update to multi dim
            x = x.reshape(-1, dims)

            #run the function without tracking gradients        
            with torch.no_grad():
                out = func(t, x)
                
            #reshape back to 1-dim output
            return out.reshape(-1)
        
        return new_func