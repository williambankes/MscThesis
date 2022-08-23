# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:36:26 2022

@author: William
"""

import pytorch_lightning as pl

import wandb

import torch
import torch.nn as nn

    

if __name__ == '__main__':
    
    from AutoEncodedFlows.models.baseline_models import AELinearModel, VAELinearModel
    from AutoEncodedFlows.models.pl_learners import AELearner, VAELearner
    from AutoEncodedFlows.datasets import SCurveDataset
    from AutoEncodedFlows.utils.experiments import Experiment, ExperimentRunner
    from AutoEncodedFlows.utils.wandb_analysis import wandb_3d_point_cloud
    
    #Setup AutoEncoder Baseline
    
    trainer_args = {'gpus':1 if torch.cuda.is_available() else 0,
                    'min_epochs':200,
                    'max_epochs':200,
                    'enable_checkpointing':False}

    scurve_train_dataset_args = {'n_samples':10_000} 
    scurve_test_dataset_args = {'n_samples':512}
    AELinearModel_args = {'input_dims':3,
                          'hidden_dims':32, 
                          'latent_dims':2,
                          'latent_hidden_dims':16}
    VAELinearModel_args = AELinearModel_args
    VAELearner_args = {'latent_dims':2,
                       'input_dims':3}
    dataloader_args = {'batch_size':256,
                       'shuffle':True}
    
    
    exps = [Experiment(project='AutoEncodingFlowsSimple',
                       learner=AELearner,
                       model=AELinearModel,
                       train_dataset=SCurveDataset,
                       train_dataset_args=scurve_train_dataset_args,
                       test_dataset=SCurveDataset,
                       test_dataset_args=scurve_test_dataset_args,
                       trainer_args=trainer_args,
                       dataloader_args=dataloader_args,
                       learner_args={},
                       model_args=AELinearModel_args,                   
                       group_name=None,
                       ask_notes=False)]
    
    exps_analysis = [[wandb_3d_point_cloud]]
    
    vae_exp = [Experiment(project='AutoEncodingFlowsSimple',
                         learner=VAELearner,
                         model=VAELinearModel,
                         train_dataset=SCurveDataset,
                         train_dataset_args=scurve_train_dataset_args,
                         test_dataset=SCurveDataset,
                         test_dataset_args=scurve_test_dataset_args,
                         trainer_args=trainer_args,
                         dataloader_args=dataloader_args,
                         learner_args=VAELearner_args,
                         model_args=VAELinearModel_args,
                         group_name=None,
                         ask_notes=False)]
    exps_analysis = [[wandb_3d_point_cloud]]
      
    exps.extend(vae_exp)
    ExperimentRunner.run_experiments(exps*10, [[wandb_3d_point_cloud]]*20)
