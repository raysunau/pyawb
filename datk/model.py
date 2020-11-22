#%%
from logging import getLogger
import os
import json
import yaml
import logging
import pandas as pd

from datk.configs import configs
from datk.utils import read_yaml,create_yaml,extract_params,read_json

logger = logging.getLogger(__name__)
logging.basicConfig(
            format = '%(asctime)s:%(levelname)s:%(message)s',
            level = logging.INFO
        )

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
        self.logfile = kwargs.get('logfile')
        self.command = kwargs.get('cmd')

        if self.logfile != None:
            self._set_logger(log_file = self.logfile)
        else:
            self.logger = logging.getLogger(name=__name__)
            self.logger.setLevel(logging.INFO)

        self.logger.info(f"Entered kwargs: {kwargs}")

        if not self.command or self.command not in self.input_cmds:
            raise Exception(f"You must enter a valid command.\n"
                            f"available commands: {self.input_cmds}")

        if self.command == "fit":
            self.yml_path = kwargs.get('yaml_path')
            file_ext = self.yml_path.split('.')[-1]
            logger.info(f"You passed the configurations as a {file_ext} file.")

            self.yaml_configs = read_yaml(self.yml_path) if file_ext == 'yaml' else read_json(self.yml_path)
            logger.info(f"your chosen configuration: {self.yaml_configs}")

            # dataset options given by the user
            self.dataset_props: dict = self.yaml_configs.get('dataset', self.default_dataset_props)
            # model options given by the user
            self.model_props: dict = self.yaml_configs.get('model', self.default_model_props)
            # list of target(s) to predict
            self.target: list = self.yaml_configs.get('target')

            self.model_type: str = self.model_props.get('type')
            logger.info(f"dataset_props: {self.dataset_props} \n"
                        f"model_props: {self.model_props} \n "
                        f"target: {self.target} \n")

        # if entered command is evaluate or predict, then the pre-fitted model needs to be loaded and used
        else:
            self.model_path = kwargs.get('model_path', self.default_model_path)
            logger.info(f"path of the pre-fitted model => {self.model_path}")
            # load description file to read stored training parameters
            with open(self.description_file, 'r') as f:
                dic = json.load(f)
                self.target: list = dic.get("target")  # target to predict as a list
                self.model_type: str = dic.get("type")  # type of the model -> regression or classification
                self.dataset_props: dict = dic.get('dataset_props')  # dataset props entered while fitting
        getattr(self, self.command)()

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

    def fit(self):
        print("fit model!!")
    
    @staticmethod
    def create_init_config_file(model_type=None, model_name=None, target=None, *args, **kwargs):
        path = configs.get('init_file_path', None)
        if not path:
            raise Exception("You need to provide a path for the init file")

        dataset_props = ModelTrainer.default_dataset_props
        model_props = ModelTrainer.default_model_props
        if model_type:
            logger.info(f"user selected model type = {model_type}")
            model_props['type'] = model_type
        if model_name:
            logger.info(f"user selected algorithm = {model_name}")
            model_props['algorithm'] = model_name

        logger.info(f"initalizing a default ModelTrainer.yaml in {path}")
        default_data = {
            "dataset": dataset_props,
            "model": model_props,
            "target": ['provide your target(s) here'] if not target else [tg for tg in target.split()]
        }
        created = create_yaml(default_data, path)
        if created:
            logger.info(f"a default igel.yaml is created for you in {path}. "
                        f"you just need to overwrite the values to meet your expectations")
        else:
            logger.warning(f"something went wrong while initializing a default file")



