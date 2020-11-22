
#%%
# from .context import datk
import sys
import os.path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datk.model import ModelTrainer

#%%

def test_init():
    
    tester = ModelTrainer(df_input="test",cmd = "fit")
    assert tester.RAND_SEED == 42
    assert tester.default_dataset_props.get('split').get('test_size') == 0.2

# %%
