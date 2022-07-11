# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:23:09 2022

@author: William
"""

import wandb
import pytorch_lightning as pl
import torch.utils.data as data

class Experiment:
    
    def __init__(self, project, notes, tags, learner, model, dataset, 
                 trainer_args, model_args, dataset_args, dataloader_args):
        
        #Setup configs:
        configs = {'Learner': learner.__name__,
                   'Model'  : model.__name__,
                   'Dataset': dataset.__name__}
        configs.update(model_args)
        configs.update(trainer_args)
        configs.update(dataset_args)
        configs.update(dataloader_args)
        
        #May restructure depending on project scope:
        self.experiment_name = "{}_{}".format(configs['Model'],
                                              configs['Dataset'])
        
        print('Creating Experiment:{}'.format(self.experiment_name))
        #Setup model and trainer:
        self.runner = wandb.init(
                        project=project,
                        name=self.experiment_name,
                        notes=notes,
                        tags=tags,
                        config=configs)
                
        self.model = model(**model_args)
        self.learner = learner(self.model) #Can add learner_args...
        self.trainer = pl.Trainer(**trainer_args)
        self.dataloader = data.DataLoader(dataset(**dataset_args),
                                          **dataloader_args)
                    
    def run(self):
        
        self.trainer.fit(self.learner, train_dataloaders=self.dataloader)
        
            
    def wandb_analyse(self, analyse_funcs):
        
        for i, func in enumerate(analyse_funcs):
            output = func(self.trainer.model, self.dataloader)
            self.runner.log(output)
            
    def analyse(self, analyse_funcs):
        
        for i, func in enumerate(analyse_funcs):
            output = func(self.trainer.model, self.dataloader)
            
    def finish(self):
        print('Experiment:{} finishing'.format(self.experiment_name))
        self.runner.finish()
        
        
def get_experiment_notes():
    
    notes = input("Experiment Note:")
    
    #assertions here
    
    return notes