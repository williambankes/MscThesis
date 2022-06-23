# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:34:50 2022

@author: William
"""

#%% Imports

from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn import *
from torchdyn.nn import Augmenter
from torchdyn.utils import *
from torchdyn.models import CNF, hutch_trace

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.distributions import MultivariateNormal
import pytorch_lightning as pl

torch.set_num_threads(16)

#%% TorchDyn CNF implementation:
d = ToyDataset()
n_samples = 1 << 14
n_gaussians = 7

X, yn = d.generate(n_samples, 'diffeqml', noise=5e-2)
X = (X - X.mean())/X.std()

plt.figure(figsize=(3, 3))
plt.scatter(X[:,0], X[:,1], c='blue', alpha=0.3, s=1)

#device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = torch.Tensor(X)
train = data.TensorDataset(X_train)
trainloader = data.DataLoader(train, batch_size=1024, shuffle=True)    
   

f = nn.Sequential(
        nn.Linear(2, 64),
        nn.Softplus(),
        nn.Linear(64, 64),
        nn.Softplus(),
        nn.Linear(64, 64),
        nn.Softplus(),
        nn.Linear(64, 2),
    )

prior = MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device)) 

# stochastic estimators require a definition of a distribution where "noise" vectors are sampled from
noise_dist = MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))

# cnf wraps the net as with other energy models
cnf = CNF(f, noise_dist=noise_dist, trace_estimator=hutch_trace)
nde = NeuralODE(cnf, solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4)

#Augments the model dimensions to allow for the trace component: -> fix this
model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1),
                      nde)

#%%
class Learner(pl.LightningModule):
    """
    Data and models are correctly managed by pl but prior distribution defined
    outside the class isn't look into defining models within Learner to correct
    """    
    
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
        self.iters = 0
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        self.iters += 1 
        x = batch[0] 
        t_eval, xtrJ = self.model(x)
        xtrJ = xtrJ[-1]
                
        logprob = prior.log_prob(xtrJ[:,1:]).to(x) - xtrJ[:,0]
        loss = -torch.mean(logprob)
        nde.nfe = 0
        return {'loss': loss}   
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-5)

    def train_dataloader(self):
        return trainloader
    
learn = Learner(model)
trainer = pl.Trainer(gpus=1, max_epochs=500)
trainer.fit(learn);

#%% Visualise the Sample:
    
sample = prior.sample(torch.Size([1 << 14]))
# integrating from 1 to 0
model[1].s_span = torch.linspace(1, 0, 2)
model = model.to(device)
_, new_x = model(sample)
new_x = new_x.cpu().detach()[-1]

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.scatter(new_x[:,1], new_x[:,2], s=2.3, alpha=0.2, linewidths=0.1, c='blue', edgecolors='black')
plt.xlim(-2, 3)
plt.ylim(-2, 2)

plt.subplot(122)
plt.scatter(X_train[:,0], X_train[:,1], s=3.3, alpha=0.2, c='red',  linewidths=0.1, edgecolors='black')
plt.xlim(-2, 3)
plt.ylim(-2, 2)


#%% Create Gif of the trajectory














