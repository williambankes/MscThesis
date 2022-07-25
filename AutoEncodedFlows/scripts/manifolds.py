# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:12:10 2022

@author: William
"""
import wandb
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torchdyn.core import NeuralODE
from torchdyn.models import CNF, hutch_trace
from torchdyn.nn import Augmenter, DepthCat
import pytorch_lightning as pl
from AutoEncodedFlows.utils.experiments import Experiment
from AutoEncodedFlows.datasets import Manifold1DDatasetNoise


class CNFLearner(pl.LightningModule):
    
    def __init__(self, vector_field:nn.Module, dims:int):
        """
        Learner setup for Torchdyn CNF model

        Parameters
        ----------
        model : nn.Module
            AutoEncoder model with encode and decode methods

        Returns
        -------
        None.

        """
        
        super().__init__()
        self.__name__ = 'CNFLearner'
        self.iters = 0
        self.dims = dims
        self.losses = list()
        
        #Define model parameters:
        ode_solver_args = {'solver':'tsit5'}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #Create Jacobian noise dist and base dist:
        self.mean = torch.zeros(self.dims).to(device)
        self.cov = torch.eye(self.dims).to(device)
        self.base_dist = MultivariateNormal(self.mean, self.cov)
        
        #Create model:
        cnf = CNF(vector_field,
                  noise_dist=self.base_dist,
                  trace_estimator=hutch_trace)
        node = NeuralODE(cnf, **ode_solver_args)
        self.model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1),
                                   node)
        wandb.watch(self.model)
        
        
    def forward(self, x):
        
        #Set model time span forward:
        self.model[1].t_span = torch.linspace(0, 1, 10)
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        self.iters += 1 
        t_eval, xtrJ = self.model(batch)
        xtrJ = xtrJ[-1] #select the end point of the trajectory:
        logprob = self.base_dist.log_prob(xtrJ[:,1:]).to(batch) - xtrJ[:,0]
        loss = -torch.mean(logprob)
        
        if self.current_epoch % 10 == 0:
            self.losses.append(loss.cpu().detach())
        
        #wandb logging:
        wandb.log({'training loss': loss.detach().item(),
                   'epoch': self.current_epoch})
           
        return {'loss': loss}   
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=2e-3, weight_decay=1e-5)

class VectorFieldNoTime(nn.Module):
    
    def __init__(self, dims, hidden_dims):
        
        super().__init__()
        
        self.__name__ = 'cnf_vector_field'
        self.network = nn.Sequential(
                           nn.Linear(dims, hidden_dims),
                           nn.Tanh(),
                           nn.Linear(hidden_dims, hidden_dims),
                           nn.Tanh(),
                           nn.Linear(hidden_dims, hidden_dims),
                           nn.Tanh(),
                           nn.Linear(hidden_dims, dims))
    
    def forward(self, x):
        return self.network(x)
    
class VectorFieldTime(nn.Module):
    
    def __init__(self, dims, hidden_dims):
        
        super().__init__()
        
        self.__name__ ='cnf_vector_field_w_time'
        self.network = nn.Sequential(
                           DepthCat(1),
                           nn.Linear(dims + 1, hidden_dims),
                           nn.Tanh(),
                           DepthCat(1),
                           nn.Linear(hidden_dims + 1, hidden_dims),
                           nn.Tanh(),
                           DepthCat(1),
                           nn.Linear(hidden_dims + 1, hidden_dims),
                           nn.Tanh(),
                           nn.Linear(hidden_dims, dims))
        
    def forward(self, x):
        return self.network(x)
        

def wandb_manifold1D_scatter_plot(model, dataloader):
    
    torch.cuda.empty_cache()
    
    #Get Data:
    dataset = dataloader.dataset
    data = dataset.get_dataset()    
    
    #Move Data and model onto GPU (due to distributions not having device)
    data = data.cuda() if torch.cuda.is_available() else data.cpu()    
    model = model.cuda() if torch.cuda.is_available() else model
    
    #Process trained model:
    _, output = model(data)
    output = output[-1,:,1:].cpu().detach().numpy()
    
    #Create wandb log from data:
    table = wandb.Table(data=output, columns=['x', 'y'])
    return {"test graph" : wandb.plot.scatter(table, 'x', 'y', title="CNF Transform")}

               
if __name__ == '__main__':
    
    import sys
    
    #Wrap into config file or command line params
    if '--test' in sys.argv: test=False #if --test then test=False
    else: test=True
    n_iters = 10            
    trainer_args = {'gpus':1 if torch.cuda.is_available() else 0,
                    'min_epochs':100 if test else 1,
                    'max_epochs':100 if test else 1,
                    'enable_checkpointing':False}
    learner_args = {'dims':2}
    model_args = {'dims':2,
                  'hidden_dims':64}
    dataset_args = {'n_samples':10_000,
                    'noise':0.15}
    dataloader_args = {'batch_size':508,
                       'shuffle':True}
    
    #Wrap multiple runs into Experiment Runner? -> probably
    #Check if test run:
    
    for n in range(n_iters):
        exp = Experiment(project='1DManifoldExperiments',
                          tags=['MscThesis', 'CNF', 'Noise=0.1'],
                          learner=CNFLearner,
                          model=VectorFieldTime,
                          dataset=Manifold1DDatasetNoise,
                          trainer_args=trainer_args,
                          learner_args=learner_args,
                          model_args=model_args,
                          dataset_args=dataset_args,
                          dataloader_args=dataloader_args,
                          group_name=None if test else "Test_Run",
                          experiment_name="{}".format(n),
                          ask_notes=False)
        
        #Try catch to ensure wandb.finish() is called:
        try:
            exp.run()
            exp.wandb_analyse([wandb_manifold1D_scatter_plot])
        finally:
            exp.finish()
