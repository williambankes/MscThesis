# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:12:10 2022

@author: William
"""

import torch
from AutoEncodedFlows.utils.experiments import Experiment
from AutoEncodedFlows.datasets import Manifold1DDatasetNoise
from AutoEncodedFlows.models.manifold_models import VectorFieldMasked, VectorFieldTime
from AutoEncodedFlows.models.manifold_models import CNFLearner, MaskedCNFLearner
from AutoEncodedFlows.utils.wandb_analysis import wandb_manifold1D_scatter_plot
                
                   
if __name__ == '__main__':
    
    import sys
    
    #Wrap into config file or command line params
    if '--test' in sys.argv: test=False #if --test then test=False
    else: test=True
                
    test=False
    
    n_iters = 1
    trainer_args = {'gpus':1 if torch.cuda.is_available() else 0,
                    'min_epochs':20 if test else 1,
                    'max_epochs':100 if test else 1,
                    'enable_checkpointing':False,
                    'check_val_every_n_epoch':5}
    learner_args = {'dims':2}
    model_args = {'dims':2,
                  'hidden_dims':64}
    dataset_args = {'n_samples':10000,
                    'noise':0}
    test_dataset_args = dataset_args
    dataloader_args = {'batch_size':256,
                       'shuffle':True}
    early_stopping_args = {'monitor':'val_loss',
			               'patience':2,
                           'mode':'min'}    
    #Wrap multiple runs into Experiment Runner? -> probably
    #Check if test run:
    
    for n in range(n_iters):
        exp = Experiment(project='1DManifoldExperiments',
                          tags=['MscThesis', 'CNF', 'Noise=0'],
                          learner=MaskedCNFLearner,
                          model=VectorFieldMasked,
                          dataset=Manifold1DDatasetNoise,
                          test_dataset=Manifold1DDatasetNoise,
                          test_dataset_args=test_dataset_args,
                          trainer_args=trainer_args,
                          learner_args=learner_args,
                          model_args=model_args,
                          dataset_args=dataset_args,
                          dataloader_args=dataloader_args,
                          early_stopping_args=early_stopping_args,
                          group_name=None if test else "Test_Run",
                          experiment_name="{}".format(n),
                          ask_notes=False)
        
        #Try catch to ensure wandb.finish() is called:
        try:
            exp.run()
            with torch.no_grad():
                exp.wandb_analyse([wandb_manifold1D_scatter_plot])
        finally:
            exp.finish()
