# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:24:24 2022

@author: William
"""

#%% imports
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from NODE.node import ODEF, NeuralODE
from sklearn.datasets import make_moons


#%% Make dataset

#Generate a dataset:
data, labels = make_moons(n_samples=100, noise=0.05)
data, labels = torch.tensor(data).float(), torch.tensor(labels).reshape(-1, 1).float()

test_set, _ = make_moons(n_samples=100, noise=0.05)
test_set = torch.tensor(test_set).float()

#Plot dataset: 
fig, axs = plt.subplots()
axs.scatter(data[:,0], data[:,1], c=labels)

#%% Create Model

class ClassifierODEF(ODEF):
    """
    Network parameterises the Neural ODE classifier.
    """
    
    def __init__(self):
        super().__init__()
        
        #define network:
        self.net = nn.Sequential(
            nn.Linear(3,32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,2))
        
        
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
            nn.Softmax(dim=-1))
    
        
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

data.requires_grad = False

epochs = 400
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

#%% Visualisations

#Plot NODE Feature Space - Duh!
out1 = model(data)
out2 = model(data, visualize=True)
    
fig, axs = plt.subplots()
axs.scatter(out2[1,:,0].detach(), out2[1,:,1].detach(), c=labels)

#Gradient Field visualised

#Trajectories

#%% Trace gradients -> CNF

#Define Gradient network:
    
class CNFODEF(ODEF):
    """
    Network parameterises the Neural ODE classifier.
    """
    
    def __init__(self):
        super().__init__()
        
        #define network:
        self.net = nn.Sequential(
            nn.Linear(3,32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,2))
        
                
    def _calc_trace_dfdx(self, f, x):
        
        """Calculates the trace of the Jacobian df/dz.
        Taken from: torchdiffeq/examples/cnf.py
        """
               
        sum_diag = 0.
        for i in range(x.shape[1]):
            sum_diag += torch.autograd.grad(f[:, i].sum(), x, create_graph=True)[0][:, i]
        return sum_diag.reshape(-1, 1)
        
        
    def forward(self, t, x):
                       
        #Split input into state and logp init
        data = x[:,:-1]
        
        with torch.set_grad_enabled(True):
        
            data.requires_grad_()
            t = t.expand(x.shape[0], 1) 
            xt = torch.concat([data, t], dim=-1)        
            f = self.net(xt)
            
            dlogpdt = - self._calc_trace_dfdx(f, data)
                
        return torch.concat([f, dlogpdt], axis=-1)


class CNFTransform(nn.Module):
    
    """
    Classifier for the dataset
    """
    
    def __init__(self, visualize=False):
        super().__init__()
                        
        self.node = NeuralODE(ClassifierODEF())
        self.classifier = nn.Sequential(
            nn.Softmax(dim=-1))
    
        
    def forward(self, x, visualize=False):
        
        if visualize:
            x = self.node(x, return_whole_sequence = True)
            print(x.shape)
            return x
            
        else:
            x = self.node(x)
            return self.classifier(x)
        
cnf_model = CNFTransform()
cnf_data = torch.concat([data, torch.zeros(data.shape[0], 1)], axis=-1)

#%% Test Gradient Model:

#Test Gradient network
t0 = torch.tensor(0.)
traceModel = CNFODEF()
trace_output = traceModel(t0, cnf_data)

#Test Flow network
cnf_test_output = cnf_model(cnf_data).sum()
cnf_test_output.backward()
































