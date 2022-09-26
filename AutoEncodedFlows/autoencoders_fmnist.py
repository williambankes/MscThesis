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
    
    from AutoEncodedFlows.models.baseline_models import VAEStdConvModel, AEStdConvModel
    from AutoEncodedFlows.models.fmnist_models import AENODEConvModel, AENODEAugConvModel
    from AutoEncodedFlows.models.fmnist_models import VAENODEAugConvModel
    from AutoEncodedFlows.models.pl_learners import AELearner, VAELearner
    from AutoEncodedFlows.utils.experiments import Experiment, ExperimentRunner
    from AutoEncodedFlows.utils.wandb_analysis import wandb_image_reconstruction
    from torchvision.datasets import FashionMNIST
    from torchvision import transforms
        
    #Setup AutoEncoder Baseline
    trainer_args = {'gpus':1 if torch.cuda.is_available() else 0,
                    'min_epochs':100,
                    'max_epochs':1,
                    'enable_checkpointing':False}

    fmnist_train_dataset_args = {'root':'../',
                                 'download':True,
                                 'train':True,
                                 'transform':transforms.ToTensor()} 
    fmnist_test_dataset_args = {'root':'../',
                                'download':True,
                                'train':False,
                                'transform':transforms.ToTensor()}
    AEConvModel_args = {'kernel':5,
                        'latent_dims':64,
                        'hidden_latent_dims':256}
    VAEConvModel_args = AEConvModel_args
    AElearner_args = {'target':True,
                      'fid_score_test':True}
    VAElearner_args = {'latent_dims':10, 
                       'input_dims':[1,28,28],
                       'target':True,
                       'fid_score_test':True}
    VAENODElearner_args = {'latent_dims':128, 
                           'input_dims':[1,28,28],
                           'target':True,
                       'fid_score_test':True}
    dataloader_args = {'batch_size':256,
                       'shuffle':True}
    
        
    ae_std_exps = [Experiment(project='AutoEncodingFlowsSimple',
                       learner=AELearner,
                       model=AEStdConvModel,
                       train_dataset=FashionMNIST,
                       train_dataset_args=fmnist_train_dataset_args,
                       test_dataset=FashionMNIST,
                       test_dataset_args=fmnist_test_dataset_args,
                       trainer_args=trainer_args,
                       dataloader_args=dataloader_args,
                       learner_args=AElearner_args,
                       model_args={},                   
                       group_name=None,
                       ask_notes=False)]
        
    vae_std_exps = [Experiment(project='AutoEncodingFlowsSimple',
                       learner=VAELearner,
                       model=VAEStdConvModel,
                       train_dataset=FashionMNIST,
                       train_dataset_args=fmnist_train_dataset_args,
                       test_dataset=FashionMNIST,
                       test_dataset_args=fmnist_test_dataset_args,
                       trainer_args=trainer_args,
                       dataloader_args=dataloader_args,
                       learner_args=VAElearner_args,
                       model_args={},                   
                       group_name=None,
                       ask_notes=False)]
    
    node_ae_exps = [Experiment(project='AutoEncodingFlowsSimple',
                       learner=AELearner,
                       model=AENODEAugConvModel,
                       train_dataset=FashionMNIST,
                       train_dataset_args=fmnist_train_dataset_args,
                       test_dataset=FashionMNIST,
                       test_dataset_args=fmnist_test_dataset_args,
                       trainer_args=trainer_args,
                       dataloader_args=dataloader_args,
                       learner_args=AElearner_args,
                       model_args={'kernel':5},                   
                       group_name=None,
                       ask_notes=False)]
    
    node_vae_exps = [Experiment(project='AutoEncodingFlowsSimple',
                       learner=VAELearner,
                       model=VAENODEAugConvModel,
                       train_dataset=FashionMNIST,
                       train_dataset_args=fmnist_train_dataset_args,
                       test_dataset=FashionMNIST,
                       test_dataset_args=fmnist_test_dataset_args,
                       trainer_args=trainer_args,
                       dataloader_args=dataloader_args,
                       learner_args=VAENODElearner_args,
                       model_args={'kernel':5},                   
                       group_name=None,
                       ask_notes=False)]
   
    #ae std with fid loss - running
    #vae std with fid loss - running 
    #node with aug ae loss - setup/test
    #node with aug vae loss - setup/test
    #node without aug ae loss 
    #node without aug reduced    
        
    exps = node_ae_exps*1
    exps.extend(node_vae_exps*1)    
    #exps.extend(node_ae_exps*10)
    #exps.extend(node_vae_exps*10)
    
    ExperimentRunner.run_experiments(exps, [[wandb_image_reconstruction]]*len(exps))
