# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:24:11 2022

@author: William
"""

#%% imports

import torch
import numpy as np
import matplotlib.pyplot as plt
from NODE.ode_solver import scipySolver
from NODE.node import ode_solve

#%% Test ode_solver function: NODE code implementation

func = lambda t, x: np.exp(-1 * t) * np.ones_like(x)

batch = 5
z0 = torch.tensor(np.random.rand(batch, 1))
t0 = torch.tensor(0.)
t1 = torch.arange(0,10,1)

#Simple implementation
outputs = [ode_solve(z0, t0, t, func) for t in t1]
outputs = np.hstack(outputs)

fig, axs = plt.subplots()
for i in range(outputs.shape[0]):
    axs.plot(outputs[i,:], c='r')

#Scipy wrapper
output = scipySolver.integrate([0, 10], z0, 1, func, t_eval=np.linspace(0, 10, 500))

for i in range(output[-1]['y'].shape[0]):
    axs.plot(output[-1]['t'], output[-1]['y'][i,:], c='b')
    axs.scatter(10.0*np.ones_like(output[-1]['y'][:,-1]), #pick out last point
                output[-1]['y'][:,-1], c='b', marker="x")
    
#True Solution:
intercept = z0 + 1
for i in intercept:
    axs.plot(t1, i - torch.exp(-t1), c='g')