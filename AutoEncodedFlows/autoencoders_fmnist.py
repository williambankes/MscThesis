# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:50:22 2022

@author: William
"""

import pytorch_lightning as pl
import wandb
import torch
import torch.nn as nn

    

if __name__ == '__main__':
    
    from AutoEncodedFlows.models.baseline_models import VAEConvModel, AEConvModel
    from AutoEncodedFlows.models.models import AENODEConvModel
    from AutoEncodedFlows.models.pl_learners import AELearner, VAELearner
    from AutoEncodedFlows.utils.experiments import Experiment, ExperimentRunner
    from AutoEncodedFlows.utils.wandb_analysis import wandb_image_reconstruction
    from torchvision.datasets import FashionMNIST
    from torchvision import transforms
        
    #Setup AutoEncoder Baseline
    trainer_args = {'gpus':1 if torch.cuda.is_available() else 0,
                    'min_epochs':200,
                    'max_epochs':200,
                    'enable_checkpointing':False}

    fmnist_train_dataset_args = {'root':'../',
                                 'download':True,
                                 'train':True,
                                 'transform':transforms.ToTensor()} 
    fmnist_test_dataset_args = {'root':'../',
                                'download':True,
                                'train':False,
                                'transform':transforms.ToTensor()}
    AEConvModel_args = {'kernel':5}
    VAEConvModel_args = AEConvModel_args
    VAElearner_args = {'latent_dims':10, 
                       'input_dims':[1,28,28],
                       'target':True}
    dataloader_args = {'batch_size':256,
                       'shuffle':True}
    
    
    exps = [Experiment(project='AutoEncodingFlowsSimple',
                       learner=AELearner,
                       model=AEConvModel,
                       train_dataset=FashionMNIST,
                       train_dataset_args=fmnist_train_dataset_args,
                       test_dataset=FashionMNIST,
                       test_dataset_args=fmnist_test_dataset_args,
                       trainer_args=trainer_args,
                       dataloader_args=dataloader_args,
                       learner_args={'target':True},
                       model_args=AEConvModel_args,                   
                       group_name=None,
                       ask_notes=False)]
    
    vae_exps = [Experiment(project='AutoEncodingFlowsSimple',
                       learner=VAELearner,
                       model=VAEConvModel,
                       train_dataset=FashionMNIST,
                       train_dataset_args=fmnist_train_dataset_args,
                       test_dataset=FashionMNIST,
                       test_dataset_args=fmnist_test_dataset_args,
                       trainer_args=trainer_args,
                       dataloader_args=dataloader_args,
                       learner_args=VAElearner_args,
                       model_args=AEConvModel_args,                   
                       group_name=None,
                       ask_notes=False)]
    
    node_exps = [Experiment(project='AutoEncodingFlowsSimple',
                       learner=AELearner,
                       model=AENODEConvModel,
                       train_dataset=FashionMNIST,
                       train_dataset_args=fmnist_train_dataset_args,
                       test_dataset=FashionMNIST,
                       test_dataset_args=fmnist_test_dataset_args,
                       trainer_args=trainer_args,
                       dataloader_args=dataloader_args,
                       learner_args={'target':True},
                       model_args=AEConvModel_args,                   
                       group_name=None,
                       ask_notes=False)]
        
          
    #exps = exps*10
    #exps.extend(vae_exps*10)
    #exps.extend(node_exps*10)
    exps = vae_exps
    
    ExperimentRunner.run_experiments(exps, [[wandb_image_reconstruction]]*1)
