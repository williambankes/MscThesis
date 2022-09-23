# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:24:45 2022

@author: William
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:03:26 2022

@author: William
"""

import torch
import torch.utils.data as data

import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.distributions import MultivariateNormal
from AutoEncodedFlows.datasets import TwoMoonDataset, Manifold1DDatasetNoise

from NormalisingFlows.transforms import PlanarTransform
from NormalisingFlows.normalising_flows import CompositeFlow
from NormalisingFlows.utils import plot_density_contours

torch.set_num_threads(16)
torch.manual_seed(1)

class PlanarLearner(pl.LightningModule):
    
    def __init__(self, flow, dims:int):

        super().__init__()
        self.model = flow
        self.dims = dims
        self.losses = list()
        
        self.dist_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.mean = torch.zeros(self.dims).to(self.dist_device)
        self.cov = torch.eye(self.dims).to(self.dist_device)
        self.base_dist = MultivariateNormal(self.mean, self.cov)
        
    def forward(self, x):
        return self.flow.tforward(x)

    def training_step(self, batch, batch_idx):
        
        u, log_detJ = self.model.treverse(batch)
        
        log_probu = self.base_dist.log_prob(u)
        kl_div = -1 * (log_probu + log_detJ).mean()
        
        self.losses.append(kl_div.cpu().detach().numpy())
        
        return {'loss': kl_div}
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=2e-3, weight_decay=1e-5)

    def density_estimation_reverse(self, x):
                
        self.model.cuda() #bit hacky but works
        u, logdet = self.model.treverse(x.to(self.dist_device))
        return (self.base_dist.log_prob(u) + logdet).cpu().detach().numpy()


def visualisations(loss, nf, name):
            
    fig, axs = plt.subplots(figsize=(10,7))
    axs.plot(loss)
    axs.set_title('TwoMoonsTests.{} training loss'.format(name))

    # Plot actual and 'found' distributions:
    output_fig = plot_density_contours(lambda x: np.exp(nf.density_estimation_reverse(x)),
                          '')
    
    return output_fig


if __name__ == '__main__':

    
    trainloader = data.DataLoader(TwoMoonDataset(n_samples=1<<14, noise=0.07),
                                  batch_size=1024, shuffle=True)
    
    trainloader = data.DataLoader(Manifold1DDatasetNoise(n_samples=1<<14, noise=0.07),
                                  batch_size=1024, shuffle=True)
    flow_model = CompositeFlow(dims=2, transform=PlanarTransform,
                                num=16)
        
    data_points = list()
    
    for _ in range(10):
        
        learn = PlanarLearner(flow_model, 2)
        trainer = pl.Trainer(gpus=1, min_epochs=400, max_epochs=600)
        trainer.fit(learn, train_dataloaders=trainloader)
        
        data_points.append(learn.losses[-1])
        torch.cuda.empty_cache()
        
        visualisations(learn.losses, trainer.model, '')
        
    print(np.mean(data_points))
    print(np.std(data_points))
