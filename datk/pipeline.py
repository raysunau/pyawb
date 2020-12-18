
from sklearn.pipeline import FeatureUnion, Pipeline
import pandas as pd
import numpy as np
from typing import List,Tuple

def extract_feature_names(model, name) -> List[str]:
  """Extracts the feature names from arbitrary sklearn models
  
  Args:
    model: The Sklearn model, transformer, clustering algorithm, etc. which we want to get named features for.
    name: The name of the current step in the pipeline we are at.
    
  Returns:
    The list of feature names.  If the model does not have named features it constructs feature names
    by appending an index to the provided name.
  """

    if hasattr(model, "get_feature_names"):
        return model.get_feature_names()
    elif hasattr(model, "n_clusters"):
        return [f"{name}_{x}" for x in range(model.n_clusters)]
    elif hasattr(model, "n_components"):
        return [f"{name}_{x}" for x in range(model.n_components)]
    elif hasattr(model, "components_"):
        n_components = model.components_.shape[0]
        return [f"{name}_{x}" for x in range(n_components)]
    elif hasattr(model, "classes_"):
        return classes_
    else:
        return [name]