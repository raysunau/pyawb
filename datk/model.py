#%%
import os
import io
import yaml

import pandas as pd


# %%
class ModelTrainer:
    """
    To train a merchine learning model based on the input yaml config
    """
    RAND_SEED = 42
    
    def __init__(self
                ,df_input
                ,y=None
                ,yaml_config=None) -> None:
        
        self.config = {}
        self.df_input = df_input
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
    def _create_model(self, **kwargs):
        pass
    
    def _save_model(self,model):
        pass

    def _load_model(self):
        pass



