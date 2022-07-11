# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 15:34:05 2022

@author: William

Test how dynamically created gradient networks affect the training of torchdyn
models in pytorch lightning

This works fine in this setup but doesn't run in the full model design
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

from torchdyn.datasets import ToyDataset
from torchdyn.core import NeuralODE

#Define two Gradient Networks:
class GradNet1(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        dims = 2
        hidden_dim = 8
        
        self.network = nn.Sequential(
                    nn.Linear(dims, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Linear(hidden_dim, dims))
    
    def forward(self, x):
        return self.network(x)

class GradNet2(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        dims = 2
        hidden_dim = 8
        n_layers = 2
        
        #Create init layer:
        layers = self._create_layer(dims, hidden_dim)
        
        #Create layers:
        for l in range(n_layers):
            layers.extend(self._create_layer(hidden_dim, hidden_dim))
        
        #final layer:
        layers.append(nn.Linear(hidden_dim, dims))
        
        self.network = nn.Sequential(*layers)   
        
    def _create_layer(self, dims_in, dims_out):
        
        return [nn.Linear(dims_in, dims_out),
                nn.BatchNorm1d(dims_out),
                nn.LeakyReLU(0.01)] 

    def forward(self, x):
        return self.network(x)
           
#Pytorch Lightning Learner Class:
class Learner(pl.LightningModule):
    def __init__(self, t_span:torch.Tensor, model:nn.Module, dataloader:DataLoader):
        super().__init__()
        self.model, self.t_span = model, t_span
        self.dataloader = dataloader

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x, self.t_span)
        y_hat = y_hat[-1] # select last point of solution trajectory
        loss = nn.CrossEntropyLoss()(y_hat, y)     
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return self.dataloader

#%%

#Define Dataset:
d = ToyDataset()
X, yn = d.generate(n_samples=512, noise=1e-1, dataset_type='moons')

#Create Data loaded
X_train = torch.Tensor(X)
y_train = torch.LongTensor(yn.long())
train =   TensorDataset(X_train, y_train)
trainloader = DataLoader(train, batch_size=len(X), shuffle=True)

#Define Model:
grad_net_1 = GradNet1()
grad_net_2 = GradNet2()
print(grad_net_1, grad_net_2)
model1 = NeuralODE(grad_net_1, sensitivity='adjoint', solver='dopri5')
model2 = NeuralODE(grad_net_2, sensitivity='adjoint', solver='dopri5')

#%% Training 1:
    
learn = Learner(torch.linspace(0, 1, 5), model1, trainloader)
trainer = pl.Trainer(max_epochs=10, gpus=1)
trainer.fit(learn)

#%% Training 2:

learn = Learner(torch.linspace(0, 1, 5), model2, trainloader)
trainer = pl.Trainer(max_epochs=10, gpus=1)
trainer.fit(learn)


#%% Maybe proper initialisation will help

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)
    

