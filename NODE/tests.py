# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 15:27:58 2022

@author: William
"""

#%% imports

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from NODE.ode_solver import scipySolver
from NODE.node import ode_solve, ODEF, NeuralODE


from sklearn.datasets import make_moons


#%% Test ode_solver function 

func = lambda t, x: np.exp(-1 * t) * np.ones_like(x)

batch = 5
z0 = torch.tensor(np.random.rand(batch, 1))
t0 = torch.tensor(0.)
t1 = torch.arange(0,100,1)

ys = list()

outputs = [ode_solve(z0, t0, t, func) for t in t1]
outputs = np.hstack(outputs)

fig, axs = plt.subplots()
for i in range(outputs.shape[0]):
    axs.plot(outputs[i,:])


#%% Test ode_solver class

#define our simple grad function:
batch = 10
func = lambda t, x: np.exp(-1 * t) * np.ones_like(x)

#initial points:
x_init = np.random.rand(batch, 1)
    
#ODE solver -> runs in batch:
solver = scipySolver(func, 1)
output = solver.integrate([0, 10], x_init, t_eval=np.linspace(0, 10, 100))

fig, axs = plt.subplots()
for i in range(output[-1]['y'].shape[0]):
    axs.plot(output[-1]['y'][i,:])
    
#%% Test the NODE implementation:
    
#Generate a dataset:
data, labels = make_moons(n_samples=100, noise=0.05)
data, labels = torch.tensor(data).float(), torch.tensor(labels).reshape(-1, 1).float()

test_set, _ = make_moons(n_samples=100, noise=0.05)
test_set = torch.tensor(test_set).float()

#Plot dataset: 
fig, axs = plt.subplots()
axs.scatter(data[:,0], data[:,1], c=labels)


#%% Create model:
    #Create ODEF network that params the ODE
    #Pass this to NeuralODE class that wraps the Adjoint method function
    #Create broader classifier for the model
        
class ClassifierODEF(ODEF):
    """
    Network parameterises the Neural ODE classifier.
    """
    
    def __init__(self):
        super().__init__()
        
        #define network:
        self.net = nn.Sequential(
            nn.Linear(3,10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,2))
        
        
    def forward(self, x, t):
        #Takes x, t
                
        #change t dims to concat...
        t = t.expand(x.shape[0], 1)       
        x = torch.cat([x, t], dim=-1)
        return self.net(x)
        
    
        
class MoonsClassifier(nn.Module):
    
    """
    Classifier for the dataset
    """
    
    def __init__(self):
        super().__init__()
        
        self.node = NeuralODE(ClassifierODEF())
        self.classifier = nn.Sequential(
            nn.Softmax())
        
    def forward(self, x):
        
        x = self.node(x)
        return self.classifier(x)
    
    
#%% Training

model = MoonsClassifier()    
epochs = 1000
losses = list()

#labels= nn.functional.one_hot(labels.reshape(-1).to(torch.int64))

optim = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    
    optim.zero_grad()
    
    #calculate output:
    output = model(data)
    loss = nn.NLLLoss()(output, labels.reshape(-1).to(torch.int64))
    
    #gradient update:
    loss.backward()
    optim.step()
    
    losses.append(loss.detach().item())
    
#plot training losses:
fig, axs = plt.subplots()
axs.plot(losses)

#Plot labelled data:
preds = model(test_set)
preds = torch.where(preds[:,0] > 0.5, 1, 0)

fig, axs = plt.subplots()
axs.scatter(test_set[:,0], test_set[:,1], c=preds)
    
    
    
    







        
        
        
        
    
    



        