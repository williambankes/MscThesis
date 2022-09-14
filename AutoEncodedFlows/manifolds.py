# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:12:10 2022

@author: William
"""

import torch
from AutoEncodedFlows.utils.experiments import Experiment, ExperimentRunner
from AutoEncodedFlows.datasets import Manifold1DDatasetNoise
from AutoEncodedFlows.models.manifold_models import VectorFieldMasked, VectorFieldTime, VectorFieldNoTime
from AutoEncodedFlows.models.manifold_models import CNFLearner, MaskedCNFLearner
from AutoEncodedFlows.utils.wandb_analysis import wandb_manifold1D_sample_scatter_plot
                
                   
if __name__ == '__main__':
    
    import sys
    
    #Wrap into config file or command line params
    if '--test' in sys.argv: test=False #if --test then test=False
    else: test=True
                
    test=False

    trainer_args = {'gpus':1 if torch.cuda.is_available() else 0,
                    'min_epochs':20 if test else 1,
                    'max_epochs':100 if test else 1,
                    'enable_checkpointing':False,
                    'check_val_every_n_epoch':5}
    learner_args = {'dims':2}
    model_args = {'dims':2,
                  'hidden_dims':64}
    train_dataset_args = {'n_samples':10000,
                          'noise':0}
    test_dataset_args = train_dataset_args
    test_dataset_args = train_dataset_args
    dataloader_args = {'batch_size':256,
                       'shuffle':True}
    early_stopping_args = {'monitor':'val_loss',
			               'patience':2,
                           'mode':'min'}    
    

    exps = [Experiment(project='1DManifoldExperiments',
                      learner=CNFLearner,
                      model=VectorFieldNoTime,
                      train_dataset=Manifold1DDatasetNoise,
                      train_dataset_args=train_dataset_args,
                      test_dataset=Manifold1DDatasetNoise,
                      test_dataset_args=test_dataset_args,
                      trainer_args=trainer_args,
                      dataloader_args=dataloader_args,
                      learner_args=learner_args,
                      model_args=model_args,
                      group_name="Local Test Runs",
                      ask_notes=False)]
                          
    ExperimentRunner.run_experiments(exps,
                    [[wandb_manifold1D_sample_scatter_plot]]*len(exps))
