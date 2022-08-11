# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:23:09 2022

@author: William
"""

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.utils.data as data


class Experiment:
    
    def __init__(self, project, tags, learner, model, dataset, 
                 trainer_args, learner_args, model_args, dataset_args,
                 dataloader_args, test_dataset = None, test_dataset_args=None , 
                 early_stopping_args=None, group_name=None,
                 experiment_name=None, ask_notes=True):
        """
        A class to manage wandb api interactions and pytorch lightning training
        interactions. Input relevant models and parameters then run the .run()
        method to run pytorch lightning model training.

        Parameters
        ----------
        project : str
            str with the wandb project name 
        tags : list()
            A list of short descriptors for each experiment
        learner : <pytorchlightning.LightningModule>
            A LightningModule implementation that manages the training of the
            model
        model : <pytorch.nn.Module>
            The pytorch model that is given as an argument to the trainer. 
        dataset : <pytorch.utils.data.Dataset>
            Dataset on which the model training is run
        <param>_args : dict
            Dictionary of arguments to be passed to the relevant parameter. These
            are also logged in the wandb config so should be a python primative.
	early_stopping : int
	    If not entered early stopping will not be applied to the model, if int
	    the patience of the callback will be set as early_stopping.

        Returns
        -------
        None.

        """
        
        #### Wrap init code into param processing functions
        
        #Setup config file for wandb:
        configs = {'Learner': learner.__name__,
                   'Model'  : model.__name__,
                   'Dataset': dataset.__name__}
        configs.update(model_args)
        configs.update(learner_args)
        configs.update(trainer_args)
        configs.update(dataset_args)
        configs.update(dataloader_args)
        
        #Setup wandb group name, experiment_name and note
        if group_name is None: self.group_name = "{}_{}_{}".format(configs['Learner'],
                                                                   configs['Model'],
                                                                   configs['Dataset'])
        else:                  self.group_name = group_name
        
        if experiment_name is None: self.experiment_name = get_user_input("Enter Experiment Name:")
        else:                       self.experiment_name = experiment_name 
        
        if ask_notes: notes = get_user_input("Enter notes on experiment {}:".\
                                             format(self.experiment_name))
        else:         notes = "N/A"
        print('Creating Experiment:{} in group: {}'.format(self.experiment_name,
                                                           self.group_name))

        #Setup wandb experiment:
        self.runner = wandb.init(
                        project=project,
                        group=self.group_name,
                        name=self.experiment_name,
                        notes=notes,
                        tags=tags,
                        config=configs)

        #Setup early stopping args:
        if early_stopping_args is None: early_stopping_callback = None
        else: early_stopping_callback = EarlyStopping(**early_stopping_args)
        self.early_stopping = early_stopping_args #replace with val dataloader setup     
        
        #Setup Test dataset 
        if test_dataset is None:
            if test_dataset_args is None: 
                print('warning: test_dataset_args set when test_dataset is None')
            self.test_dataloader = None
        else:
            assert test_dataset_args is not None,\
            'test_dataset_args must be speicified when test_dataset is not None' 
            self.test_dataloader = data.DataLoader(test_dataset(**test_dataset_args),
                                                   **dataloader_args)
        self.dataloader = data.DataLoader(dataset(**dataset_args),
                                                  **dataloader_args)
        
        
        self.model = model(**model_args)
        self.learner = learner(self.model, **learner_args)
        self.trainer = pl.Trainer(**trainer_args, callbacks=early_stopping_callback)
        self.fitted = False

                    
    def run(self):
        """
        Run the trainer .fit() method

        Returns
        -------
        None.

        """
        #Sort into init
        if self.early_stopping is None: val_dataloader = None
        else: val_dataloader = self.dataloader

        self.trainer.fit(self.learner, train_dataloaders=self.dataloader,
                                       val_dataloaders=val_dataloader)
        if self.test_dataloader is not None:
            self.trainer.test(self.learner, dataloaders=self.test_dataloader)
        
        self.fitted = True
        
            
    def wandb_analyse(self, analyse_funcs):
        """
        Run analytics functions custom to an experiment. The outputs are expected
        to be wandb logs and are processed and passed to the wandb logger
        process

        Parameters
        ----------
        analyse_funcs : <function>(model:pl.LightningModule,
                                 dataloader:pytorch.utils.data.Dataloader)
            Custom analytics function of suitable format that analyses and 
            returns a wandb log.
        
        Returns
        -------
        None.

        """
        
        for i, func in enumerate(analyse_funcs):
            output = func(self.trainer.model, self.dataloader)
            self.runner.log(output)
            
    def analyse(self, analyse_funcs):
        """
        Run general analytics functions on model and dataset.
        
        Parameters
        ----------
        analyse_funcs : <function>(model:pl.LightningModule,
                                   dataloader:pytorch.utils.data.Dataloader)
            Custom analytics function

        Returns
        -------
        None.

        """
        for i, func in enumerate(analyse_funcs):
            func(self.trainer.model, self.dataloader)
            
    def finish(self):
        """
        Ensures the wandb run is finished. More functionality can be added
        as necessary...

        Returns
        -------
        None.

        """
        print('Experiment:{} finishing'.format(self.experiment_name))
        self.runner.finish()
        
        
def get_user_input(query):
    """
    Get experiment note string via input query

    Returns
    -------
    notes : str

    """
    
    notes = input(query)

    assert isinstance(notes, str), ""    
    
    return notes

def get_user_confirmation(query):
    """

    Parameters
    ----------
    query : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    conf = input(query)
    assert isinstance(conf, str), "get_user_confirmation: conf must be of type string"
    conf = str.lower(conf)
    
    affirm = ['y', 'yes']
    
    if conf in affirm: return True
    else: return False
    
    
