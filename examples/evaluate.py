
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(''))))

from datk.model import ModelTrainer

eval_params = {
    'cmd':'evaluate',
    'data_path': './examples/train_titanic.csv'
}

ModelTrainer(**eval_params)
