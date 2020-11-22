#%%
from logging import getLogger
import os

import yaml
import logging
import pandas as pd

from datk.configs import configs


# %%
class ModelTrainer:
    """
    To train a merchine learning model based on the input yaml config
    """
    RAND_SEED = 42
    input_cmds = ('fit','evaluate','predict','experiment')
    supported_types = ('regression', 'classification', 'clustering')
    results_path = configs.get('results_path')  # path to the results folder
    default_model_path = configs.get('default_model_path')  # path to the pre-fitted model
    description_file = configs.get('description_file')  # path to the description.json file
    evaluation_file = configs.get('evaluation_file')  # path to the evaluation.json file
    prediction_file = configs.get('prediction_file')  # path to the predictions.csv
    default_dataset_props = configs.get('dataset_props')  # dataset props that can be changed from the yaml file
    default_model_props = configs.get('model_props')  # model props that can be changed from the yaml file
    model = None



    def __init__(self
                ,*args, **kwargs) -> None:
        
        self.data_path:str = kwargs.get('data_path')
        self.df_input = kwargs.get('df_input') 
        self.logfile = kwargs.get('logfile')

        if self.logfile != None:
            self._set_logger(log_file = self.logfile)
        else:
            self.logger = logging.getLogger(name=__name__)
            self.logger.setLevel(logging.INFO)
        self.logger.info(f"Entered kwargs: {kwargs}")


    def _set_logger(self,
                    log_file:str = 'Pydatoolkt_logger.log',
                    log_level = logging.INFO):
        logging.basicConfig(
            format = '%(asctime)s:%(levelname)s:%(message)s',
            filename=log_file,
            filemode='w',
            level = log_level
        )
        logging.info('Start logging to ' + log_file + ' ' + str(self.model))


    def _create_model(self, **kwargs):
        pass
    
    def _save_model(self,model):
        pass

    def _load_model(self):
        pass



