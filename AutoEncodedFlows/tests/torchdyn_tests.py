# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:23:50 2022

- Train a standard NODE model

- Train a CNF model

- Stack flows

- Plan module architecture


@author: William
"""


#%% imports
from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn import *
from torchdyn.models import CNF
from torchdyn.nn import DataControl, DepthCat, Augmenter
from torchdyn.utils import *

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.distributions import MultivariateNormal
import pytorch_lightning as pl

#%% Devices:
    
dry_run = True


#%% Create dataset:
d = ToyDataset()
X, yn = d.generate(n_samples=512, noise=1e-1, dataset_type='moons')

fig, axs = plt.subplots(figsize=(5,5))
axs.scatter(X[:,0], X[:,1], s=5, c=yn)

#Create Data loaded
X_train = torch.Tensor(X)
y_train = torch.LongTensor(yn.long())
train = data.TensorDataset(X_train, y_train)
trainloader = data.DataLoader(train, batch_size=len(X), shuffle=True)
    

class Learner(pl.LightningModule):
    def __init__(self, t_span:torch.Tensor, model:nn.Module):
        super().__init__()
        self.model, self.t_span = model, t_span

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x, t_span)
        y_hat = y_hat[-1] # select last point of solution trajectory
        loss = nn.CrossEntropyLoss()(y_hat, y)     
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return trainloader
    
f = nn.Sequential(
        nn.Linear(2, 16),
        nn.Tanh(),
        nn.Linear(16, 2)
    )

t_span = torch.linspace(0, 1, 5)
model = NeuralODE(f, sensitivity='adjoint', solver='dopri5')

#%%
learn = Learner(t_span, model)
trainer = pl.Trainer(min_epochs=200, max_epochs=300)
trainer.fit(learn)

#%% Plot Trained Model

t_eval, trajectory = model(X_train, t_span)
trajectory = trajectory.detach().cpu()

color=['orange', 'blue']

fig = plt.figure(figsize=(10,2))
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
for i in range(500):
    ax0.plot(t_span, trajectory[:,i,0], color=color[int(yn[i])], alpha=.1);
    ax1.plot(t_span, trajectory[:,i,1], color=color[int(yn[i])], alpha=.1);
ax0.set_xlabel(r"$t$ [Depth]") ; ax0.set_ylabel(r"$h_0(t)$")
ax1.set_xlabel(r"$t$ [Depth]") ; ax1.set_ylabel(r"$z_1(t)$")
ax0.set_title("Dimension 0") ; ax1.set_title("Dimension 1")

#%% Plot learnt model output:
t_span, output = model(X_train)
probs = nn.Softmax(-1)(output[-1])[:,1].detach().numpy()

probs = np.where(probs > 0.5, 1, 0)



plt.scatter(X_train[:,0], X_train[:,1], c=probs)



















