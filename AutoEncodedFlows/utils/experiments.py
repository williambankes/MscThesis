# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:23:09 2022

@author: William
"""

import gc
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.utils.data as data


class ExperimentRunner:
    
    @staticmethod
    def run_experiments(exp_list, exp_analysis_list):
        """
        Run a set of experiments and their analysis functions in a batch
        run.

        Parameters
        ----------
        exp_list : <list>
            A list of instantiated experiement objects
        exp_analysis_list: <list>
            A list of analysis methods whos api is appropriate for the wandb_analysis
            and analysis methods below.
        n_iter : int, optional
            Number of iterations to run each experiment. The default is 10.

        Returns
        -------
        None.

        """
        
        assert len(exp_list) == len(exp_analysis_list),\
            'Experiment must have wandb_analysis'
        
        for i, exp in enumerate(exp_list):
                
            #Try catch to ensure wandb.finish() is called:
            try:
                #Experiment name as number
                exp.setup_run(experiment_name='{}'.format(i))
                exp.run()
                with torch.no_grad():
                    exp.wandb_analyse(exp_analysis_list[i])
            finally:
                exp.finish()
                         


class Experiment:
    
    def __init__(self, project, learner, model, train_dataset, 
                 trainer_args, learner_args, model_args, train_dataset_args,
                 dataloader_args, val_dataset=None, val_dataset_args=None,
                 test_dataset = None, test_dataset_args=None, 
                 early_stopping_args=None, group_name=None, ask_notes=True):
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
                   'Dataset': train_dataset.__name__}
        configs.update(model_args)
        configs.update(learner_args)
        configs.update(trainer_args)
        configs.update(train_dataset_args)
        configs.update(dataloader_args)
        
        #Setup wandb group name, experiment_name and note
        if group_name is None: self.group_name = "{}_{}_{}".format(configs['Learner'],
                                                                   configs['Model'],
                                                                   configs['Dataset'])
        else:                  self.group_name = group_name
        
        

        self.configs = configs
        self.project = project
        self.ask_notes = ask_notes
        
        self.early_stopping_args = early_stopping_args
        self.dataloader_args = dataloader_args   
        self.train_dataset, self.train_dataset_args = train_dataset, train_dataset_args
        self.val_dataset, self.val_dataset_args = val_dataset, val_dataset_args
        self.test_dataset, self.test_dataset_args = test_dataset, test_dataset_args
        
        self.model_class = model
        self.model_args = model_args
        self.learner_class, self.learner_args = learner, learner_args
        self.trainer_args = trainer_args
        self.setup = False
        
    def setup_run(self, experiment_name):
        """
        setup_run allows us to control when memory intensive objects such as the
        model and dataset are instantiated. This must be run before the <run>
        method

        Returns
        -------
        None.

        """
        
        if experiment_name is None: self.experiment_name = get_user_input("Enter Experiment Name:")
        else:                       self.experiment_name = experiment_name 
        
        print('Creating Experiment:{} in group: {}'.format(experiment_name,
                                                           self.group_name))
        
        if self.ask_notes: self.notes = get_user_input("Enter notes on experiment {}:".\
                                                  format(self.experiment_name))
        else:              self.notes = "N/A"
        
        #Setup wandb experiment:
        self.runner = wandb.init(
                        project=self.project,
                        group=self.group_name,
                        name=self.experiment_name,
                        notes=self.notes,
                        config=self.configs)

        #Setup early stopping args:
        if self.early_stopping_args is None: early_stopping_callback = None
        else: early_stopping_callback = EarlyStopping(**self.early_stopping_args)
       
        #Setup Dataloaders:
        self.train_dataloader = self.init_dataloader(self.dataloader_args, self.train_dataset,
                                                     self.train_dataset_args, 'train')
        self.val_dataloader = self.init_dataloader(self.dataloader_args, self.val_dataset,
                                                     self.val_dataset_args, 'val')
        self.test_dataloader = self.init_dataloader(self.dataloader_args, self.test_dataset,
                                                    self.test_dataset_args, 'test')
        
        self.model = self.model_class(**self.model_args)
        self.learner = self.learner_class(self.model, **self.learner_args)
        self.trainer = pl.Trainer(**self.trainer_args, callbacks=early_stopping_callback)
        
        self.fitted = False
        self.setup = True
        
        
    def init_dataloader(self, dataloader_args, dataset, dataset_args, name):
        """
        Setup dataloader with args and correctly setup dataset. Handles cases
        where no DataLoader is required e.g. test and validation cases

        Parameters
        ----------
        dataloader_args : <dict>
            Dictionary with arguments specific to the DataLoader
        dataset : <pytorch.utils.data.DataSet>
            Dataset to be wrapped in DataLoader 
        dataset_args : <dict>
            Arguments for instantiating the DataSet
        name : str
            Name of the dataset being instantiated e.g. train, val, test

        Returns
        -------
        dataloader : <pytorch.utils.data.DataLoader>
            DataLoader for input Dataset 

        """
        
        if dataset is None:
            if dataset_args is None: 
                print('warning: {}_dataset_args set when {}_dataset is None'.\
                      format(name, name))
            dataloader = None
        else:
            assert dataset_args is not None,\
            '{}_dataset_args must be speicified when {}_dataset is not None'.\
                format(name, name)
            dataloader = data.DataLoader(dataset(**dataset_args), **dataloader_args)
        
        return dataloader
    
    def run(self):
        """
        Run the training, validation and test loops defined in pytorch lightning

        Returns
        -------
        None.

        """
        assert self.setup, 'Must run setup_run method first'
        
        #Run Training loop
        self.trainer.fit(self.learner, train_dataloaders=self.train_dataloader,
                                       val_dataloaders=self.val_dataloader)
        
        #Potentially add self.trainer.validate() loop here...
        
        #Run Test if test dataset provided:
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
            if self.test_dataloader is None:
                output = func(self.trainer.model, self.train_dataloader)
            else:
                output = func(self.trainer.model, self.test_dataloader)
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

        if self.setup:
            #Delete datasets and models:        
            del self.train_dataloader, self.val_dataloader, self.test_dataloader
            del self.learner, self.trainer, self.model
        gc.collect()
        
        
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
    Get user confirmation via a 'y' or 'yes' type answer... Add more options

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
    
    
