
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(''))))

from datk.model import ModelTrainer

fit_params = {
    'cmd':'predict',
    'data_path': './examples/train_titanic.csv'
}

ModelTrainer(**fit_params)
