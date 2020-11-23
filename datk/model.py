#%%
from logging import getLogger
import os
import json
import yaml
import logging
import pandas as pd


from datk.configs import configs
from datk.utils import read_yaml,create_yaml,extract_params,read_json,_reshape
from datk.preprocessing import encode,normalize,handle_missing_values

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
        
        self.data_path:str = kwargs.get('data_path',None)
        self.logfile = kwargs.get('logfile',None)
        self.command = kwargs.get('cmd',None)

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
            self.yml_path = kwargs.get('yaml_path',None)
            file_ext = self.yml_path.split('.')[-1]
            logger.info(f"You passed the configurations as a {file_ext} file.")

            self.yaml_configs = read_yaml(self.yml_path) if file_ext == 'yaml' else read_json(self.yml_path)
            logger.info(f"your chosen configuration: {self.yaml_configs}")

            # dataset options given by the user
            self.dataset_props: dict = self.yaml_configs.get('dataset', self.default_dataset_props)
            # model options given by the user
            self.model_props: dict = self.yaml_configs.get('model', self.default_model_props)
            # list of target(s) to predict
            self.target: list = self.yaml_configs.get('target',None)

            self.model_type: str = self.model_props.get('type',None)
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
    
    def _prepare_clustering_data(self):
        """
        preprocess data for the clustering algorithm
        """
        return self._process_data(target='fit_cluster')

    def _prepare_predict_data(self):
        """
        preprocess predict data to get similar data to the one used when training the model
        """
        return self._process_data(target='predict')

    
    def _prepare_fit_data(self):
        return self._process_data(target='fit')

    def _prepare_eval_data(self):
        return self._process_data(target='evaluate')

    def _process_data(self, target='fit'):
        """
        read and return data as x and y
        @return: list of separate x and y
        """
        assert isinstance(self.target, list), "provide target(s) as a list in the yaml file"
        if self.model_type != "clustering":
            assert len(self.target) > 0, "please provide at least a target to predict"

        try:
            read_data_options = self.dataset_props.get('read_data_options', None)
            dataset = pd.read_csv(self.data_path) if not read_data_options else pd.read_csv(self.data_path,
                                                                                            **read_data_options)
            logger.info(f"dataset shape: {dataset.shape}")
            attributes = list(dataset.columns)
            logger.info(f"dataset attributes: {attributes}")

            # handle missing values in the dataset
            preprocess_props = self.dataset_props.get('preprocess', None)
            if preprocess_props:
                # handle encoding
                encoding = preprocess_props.get('encoding')
                if encoding:
                    encoding_type = encoding.get('type', None)
                    column = encoding.get('column', None)
                    if column in attributes:
                        dataset, classes_map = encode(df=dataset,
                                                      encoding_type=encoding_type.lower(),
                                                      column=column)
                        if classes_map:
                            self.dataset_props['label_encoding_classes'] = classes_map
                            logger.info(f"adding classes_map to dataset props: \n{classes_map}")
                        logger.info(f"shape of the dataset after encoding => {dataset.shape}")

                # preprocessing strategy: mean, median, mode etc..
                strategy = preprocess_props.get('missing_values')
                if strategy:
                    dataset = handle_missing_values(dataset,
                                                    strategy=strategy)
                    logger.info(f"shape of the dataset after handling missing values => {dataset.shape}")

            if target == 'predict' or target == 'fit_cluster':
                x = _reshape(dataset.to_numpy())
                if not preprocess_props:
                    return x
                scaling_props = preprocess_props.get('scale', None)
                if not scaling_props:
                    return x
                else:
                    scaling_method = scaling_props.get('method', None)
                    return normalize(x, method=scaling_method)

            if any(col not in attributes for col in self.target):
                raise Exception("chosen target(s) to predict must exist in the dataset")

            y = pd.concat([dataset.pop(x) for x in self.target], axis=1)
            x = _reshape(dataset.to_numpy())
            y = _reshape(y.to_numpy())
            logger.info(f"y shape: {y.shape} and x shape: {x.shape}")

            # handle data scaling
            if preprocess_props:
                scaling_props = preprocess_props.get('scale', None)
                if scaling_props:
                    scaling_method = scaling_props.get('method', None)
                    scaling_target = scaling_props.get('target', None)
                    if scaling_target == 'all':
                        x = normalize(x, method=scaling_method)
                        y = normalize(y, method=scaling_method)
                    elif scaling_target == 'inputs':
                        x = normalize(x, method=scaling_method)
                    elif scaling_target == 'outputs':
                        y = normalize(y, method=scaling_method)

            if target == 'evaluate':
                return x, y

            split_options = self.dataset_props.get('split', None)
            if not split_options:
                return x, y, None, None
            test_size = split_options.get('test_size')
            shuffle = split_options.get('shuffle')
            stratify = split_options.get('stratify')
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=test_size,
                shuffle=shuffle,
                stratify=None if not stratify or stratify.lower() == "default" else stratify)

            return x_train, y_train, x_test, y_test

        except Exception as e:
            logger.exception(f"error occured while preparing the data: {e.args}")

    
    def fit(self, **kwargs):
        """fit a model

        Raises:
            Exception: [description]
        """
        x_train = None
        y_train = None
        x_test = None
        y_test = None

        cv_result = None
        eval_result = None
        cv_params = None

        if self.model_type == 'clustering':
            x_train = self._prepare_clustering_data()
        else:
            x_train, y_train, x_test, y_test = self._prepare_fit_data()
        self.model, model_args = self._create_model(**kwargs)
        logger.info(f"executing a {self.model.__class__.__name__} algorithm...")

    
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



