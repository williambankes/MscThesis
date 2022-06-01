# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:36:46 2022

@author: William
"""

import torch
from scipy.integrate import solve_ivp

class scipySolver():

    def __init__(self, func, dims, *func_args):
        """
        Acts as our integrator in the Neural ODE implementation. Function manages
        the scipy integrator by adding handling of batchs with different initial
        points. 
        
        Mostly taken from https://github.com/rtqichen/torchdiffeq/
        
        Potential improvements: 
            - Include numba ode solver for faster processing
            
        Parameters
        ----------
        func: <method>
        
        dims: int
        
        func_args: <iterable>
    
        """        
        
        self.func = self._prep_func(func, dims)
        self.dims = dims
        self.func_args = func_args
        
        
        
    def integrate(self, t_span, x_init, t_eval=None):
        """
        Numerical Integration of self.func across some span of time and for some
        data x of dimensions [Batch x dims]
        
        t_span:list()
        Some iterable defining the bounds of the integration
        
        x_init:torch.tensor()
        [Batch x dims] tensor containing the initial points of the integration

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
        assert x_init.shape[1] == self.dims,\
            "Input tensors dims: {} don't match construct dims: {}".format(x_init.shape[1],
                                                                           self.dims)
        #record the batch size
        batch = x_init.shape[0]
            
        #Run the solver_ivp function:
        solution = solve_ivp(self.func,
                              t_span,
                              x_init.reshape(-1),
                              args=self.func_args,
                              t_eval=t_eval)
        
        #re map the 1-d solution back to input dims:
        x_out = solution['y'].reshape(-1, batch, self.dims)
        t = solution['t']
        success = solution['success']
        
        return success, x_out, t, solution

        
    def _prep_func(self, func, dims):
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