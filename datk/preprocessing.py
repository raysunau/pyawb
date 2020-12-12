from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.pipeline import make_pipeline,Pipeline,FeatureUnion
from sklearn.compose import ColumnTransformer,make_column_selector

import numpy as np
import pandas as pd
import logging

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def update_dataset_props(dataset_props: dict, default_dataset_props: dict):
    for key1 in default_dataset_props.keys():
        if key1 in dataset_props.keys():
            for key2 in default_dataset_props[key1].keys():
                if key2 in dataset_props[key1].keys():
                    default_dataset_props[key1][key2] = dataset_props[key1][key2]

    return default_dataset_props


def handle_missing_values(df, fill_value=np.nan, strategy="mean"):
    logger.info(f"Check for missing values in the dataset ...  \n"
                f"{df.isna().sum()}  \n "
                f"{'-'*100}")

    if strategy.lower() == "drop":
        return df.dropna()

    cleaner = SimpleImputer(fill_value=fill_value, strategy=strategy)
    cleaned = cleaner.fit_transform(df)

    return pd.DataFrame(cleaned, columns=df.columns)


def encode(df, encoding_type="onehotencoding", column=None):
    if not encoding_type:
        raise Exception(f"encoding type should be -> oneHotEncoding or labelEncoding")

    if encoding_type == "onehotencoding":
        logger.info(f"performing a one hot encoding ...")
        return pd.get_dummies(df, dummy_na=True), None

    elif encoding_type == "labelencoding":
        if not column:
            raise Exception("if you choose to label encode your data, "
                            "then you need to provide the column you want to encode from your dataset")
        logger.info(f"performing a label encoding ...")
        encoder = LabelEncoder()
        encoder.fit(df[column])
        classes_map = {cls: int(lbl) for (cls, lbl) in zip(encoder.classes_, encoder.transform(encoder.classes_))}
        logger.info(f"label encoding classes => {encoder.classes_}")
        logger.info(f"classes map => {classes_map}")
        df[column] = encoder.transform(df[column])
        return df, classes_map

    else:
        raise Exception(f"encoding type should be -> oneHotEncoding or labelEncoding")


def normalize(x, y=None, method='standard'):
    methods = ('minmax', 'standard')

    if method not in methods:
        raise Exception(f"Please choose one of the available scaling methods => {methods}")
    logger.info(f"performing a {method} scaling ...")
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    if not y:
        return scaler.fit_transform(X=x)
    else:
        return scaler.fit_transform(X=x, y=y)

# sunr's function
def make_df_empty(columns, dtypes, index=None):
    """[create an empty dataframe with columns and dtypes]

    Args:
        columns ([type]): [description]
        dtypes ([type]): [description]
        index ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    df = make_df_empty(['a', 'b'], dtypes=[np.int64, np.int64])    
    """
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df

def make_column_transformer(num_missing_impute_strategy='mean'):
    """[create a data preprocess pipeline using sklearn pipeline]
    Todo
    """
    num_imputer = Pipeline([
            ("imputer", SimpleImputer(strategy=num_missing_impute_strategy,add_indicator=False))
        ])
    cat_ohe = Pipeline([
        ("cat_imputer", SimpleImputer(strategy='constant',fill_value='NA')),
        ('ohe',OneHotEncoder(dtype=np.int,handle_unknown='ignore'))
        ])
    return ColumnTransformer(
        [('imp', num_imputer, make_column_selector(dtype_include=np.number)),
        ('ohe', cat_ohe, make_column_selector(dtype_include=['object','category']))
        ], remainder='passthrough'
        )
   

def get_preprocessed_df(df,transformer,fit_flag=False):
    
    if fit_flag:
        df_ = transformer.fit_transform(df)
    else:
        df_ = transformer.transform(df)
    
    num_cols = transformer.transformers_[0][2]

    cat_cols_encoded = transformer.transformers_[1][1].steps[1][1].get_feature_names(
        transformer.transformers_[1][2])
        
    return pd.DataFrame(df_,
        columns=num_cols + list(cat_cols_encoded),
        index=df.index)