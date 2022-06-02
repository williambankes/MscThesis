# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 15:27:58 2022

To do: 
    
Integrate ScipySolver:
    
Understand type conversion through NODE process -> when is conversion
occuring... Look into torchdiffeq...

Visualise Augmented NODE 

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


#%% Test ode_solver function: NODE code implementation

func = lambda t, x: np.exp(-1 * t) * np.ones_like(x)
func2 = lambda x, t: np.exp(-1 * t) * np.ones_like(x)

batch = 5
z0 = torch.tensor(np.random.rand(batch, 1))
t0 = torch.tensor(0.)
t1 = torch.arange(0,10,1)

#Simple implementation
outputs = [ode_solve(z0, t0, t, func2) for t in t1]
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
        
        
    def forward(self, t, x):
                
        t = t.expand(x.shape[0], 1) 
        x = torch.cat([x, t], dim=-1)
        return self.net(x)
        
     
class MoonsClassifier(nn.Module):
    
    """
    Classifier for the dataset
    """
    
    def __init__(self, visualize=False):
        super().__init__()
                        
        self.node = NeuralODE(ClassifierODEF())
        self.classifier = nn.Sequential(
            #nn.Linear(2,2),
            nn.Softmax(dim=-1))
        
        #self.classifier = nn.Sequential(
        #    nn.Linear(2,2),
        #    nn.Softmax(dim=-1))
        
    def forward(self, x, visualize=False):
        
        if visualize:
            x = self.node(x, return_whole_sequence = True)
            print(x.shape)
            return x
            
        else:
            x = self.node(x)
            return self.classifier(x)
    
model = MoonsClassifier()   

#%% Training

epochs = 700
losses = list()

optim = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    
    optim.zero_grad()
    
    #calculate output:
    output = model(data)
    loss = nn.NLLLoss()(output, labels.reshape(-1).to(torch.int64))
    
    #gradient update:
    loss.backward()
    optim.step()
    
    epoch_loss = loss.detach().item()
    
    if epoch % 50 == 0:
        print('Epoch: {} loss: {}'.format(epoch, epoch_loss))
    losses.append(epoch_loss)
    
#plot training losses:
fig, axs = plt.subplots()
axs.plot(losses)

#Plot labelled data:
preds = model(test_set)
preds = torch.where(preds[:,0] > 0.5, 1, 0)

fig, axs = plt.subplots()
axs.scatter(test_set[:,0], test_set[:,1], c=preds)
    
#%% Generating NODE trajectories

out1 = model(data)
out2 = model(data, visualize=True)
    
fig, axs = plt.subplots()
axs.scatter(out2[1,:,0].detach(), out2[1,:,1].detach(), c=labels)



        