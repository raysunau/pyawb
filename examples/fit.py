
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(''))))

from datk.model import ModelTrainer

""" fit_params = {
    'cmd':'fit',
    'data_path': './examples/train_titanic.csv',
    'yaml_path': './examples/model.yaml',
    'results_path': './examples'
} """


fit_params = {
    'cmd':'fit',
    'data_path': './examples/train_titanic.csv',
    'yaml_path': './examples/xgb_model_classification.yaml',
    'results_path': './examples/fit/',
    'logfile': './examples/fit/logfile.log'
}

ModelTrainer(**fit_params)
