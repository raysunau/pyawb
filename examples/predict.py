
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(''))))

from datk.model import ModelTrainer

pred_params = {
    'cmd':'predict',
    'data_path': './examples/test_titanic.csv'
}

ModelTrainer(**pred_params)
