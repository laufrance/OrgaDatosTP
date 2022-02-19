import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.feature_extraction import FeatureHasher

def one_hot_encode(df, features_to_encode):
    return pd.get_dummies(df, columns=features_to_encode, dummy_na=True, drop_first=True) 
    
def hashing_trick_encode(df, feature, n_features):
#     fh = FeatureHasher(n_features, input_type='string')
#     hashed_features = fh.fit_transform(df[feature].astype(str)).todense()
#     hashed_features = pd.DataFrame(hashed_features).add_prefix(f"{feature}_")    
#     return df.head(), pd.concat([df, hashed_features], axis=1)
    return "hola"
