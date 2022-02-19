import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
)
from sklearn.feature_extraction import FeatureHasher
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

RANDOM_STATE = 10

dtype_X = {
        "barrio": "category",
        "dia": "object",
        "direccion_viento_tarde": "category",
        "direccion_viento_temprano": "category",
        "horas_de_sol": "float32",
        "humedad_tarde": "float32",
        "humedad_temprano": "float32",
        "id": "int32",
        "llovieron_hamburguesas_hoy": "category",
        "mm_evaporados_agua": "float32",
        "mm_lluvia_dia": "float32",
        "nubosidad_tarde": "float32",
        "nubosidad_temprano": "float32",
        "presion_atmosferica_tarde": "object",
        "presion_atmosferica_temprano": "float32",
        "rafaga_viento_max_direccion": "category",
        "rafaga_viento_max_velocidad": "float32",
        "temp_max": "float32",
        "temp_min": "float32",
        "temperatura_tarde": "float32",
        "temperatura_temprano": "float32",
        "velocidad_viendo_tarde": "float32",
        "velocidad_viendo_temprano": "float32",
    }

dtype_y = {
    "id":"int32", 
    "llovieron_hamburguesas_al_dia_siguiente": "category"
}

###-----------OBTENER-DATASET-----------###
def get_X_y_from_dataset():
    X = pd.read_csv("../parte_1/features.csv", dtype=dtype_X)
    y = pd.read_csv("../parte_1/target.csv", dtype=dtype_y)

    return X, y

def get_X_private_features():
    return pd.read_csv("./private_features.csv", dtype=dtype_X)

### -------------CONVERSIÓN------------- ###

# Transforma una lista de features numéricos (float o int) a features categóricos ordinales,
# cuyas categorías son las que le pasamos a la función.
def numerical_to_ordinal_categorical(X, features_to_transform, categories):
    for f in features_to_transform:
        transformed_f = pd.Categorical(X[f], categories=categories, ordered=True)
        X[f] = transformed_f

    return X


# Extrae el mes de un feature de tipo datetime64
def extract_month_from_date(df):
    df['mes'] = pd.DatetimeIndex(df['dia']).month
    return df

# Aplica la técnica one hot (técnicamente dummy encoding) para encodear una variable categórica no ordinal.
# Devuelve el df modificado
def one_hot_encode(df, features_to_encode):
    return pd.get_dummies(df, columns=features_to_encode, dummy_na=True, drop_first=True) 

# Aplica el hashing trick al feature pasado, agregando n_features columnas al dataframe. Ideal
# para features categóricos no ordinales de alta cardinalidad. Devuelve el df modificado
def hashing_trick_encode(df, feature, n_features):
    fh = FeatureHasher(n_features, input_type='string')
    hashed_features = fh.fit_transform(df[feature].astype(str)).todense()
    hashed_features = pd.DataFrame(hashed_features).add_prefix(f"{feature}_")
    df.drop(feature, axis=1, inplace=True)
    return pd.concat([df, hashed_features], axis=1)

### -------------SELECCIÓN------------- ###

# Recibe una lista de features para dropearlos/eliminarlos del dataframe. Inplace
def drop_features(df, features_to_drop):
    df.drop(features_to_drop, axis=1, inplace=True)

# Selecciona como mínimo "min_features_to_select" features usando como estimador
# un árbol de decisión y cross validation de 3 folds. Elimina los features no seleccionados
# y devuelve el X y los features eliminados. Los datos deben ser numéricos.
def select_features_RFECV(X, y, min_features_to_select=10):
    rfe_selector = RFECV(
        estimator=DecisionTreeClassifier(),
        min_features_to_select=min_features_to_select, 
        step = 1,
        cv=3,
        verbose=2) 
    
    rfe_selector.fit(X, y)
    eliminated_features = list(X.columns[~rfe_selector.support_])
    drop_features(X, eliminated_features)
    return X, eliminated_features

### ---------REDUCCIÓN-DIMENSIONAL------###

# Aplica PCA sobre "features_to_reduce", reemplazándolos en el dataset X por 
# las n_final_features que explican la mayor varianza. Devuelve X y la varianza explicada.
def reduce_dimension_of_features(X, n_final_features, features_to_reduce, feature_names):
    pca = PCA(n_components=n_final_features)
    principalComponents = pca.fit_transform(X[features_to_reduce])
    principalDf = pd.DataFrame(data = principalComponents, columns=feature_names)

    drop_features(X, features_to_reduce)
    final_features = list(X.columns)
    final_features.extend(list(principalDf))
    
    dictio={}
    for i,f in enumerate(final_features):
        dictio[i] = f

    X.reset_index(drop=True, inplace=True)
    principalDf.reset_index(drop=True, inplace=True)
    X = pd.concat([X, principalDf], axis = 1, ignore_index=True)
    X = X.rename(columns=dictio)

    return X, pca.explained_variance_ratio_

### -------------ESCALAMIENTO------------- ###

# Escala los features numéricos pasados en una lista usando StandardScalar (haciendo que los datos
# tengan media 0 y desvío estandard 1). Devuelve el df modificado
def standard_scale(df, features_to_scale):
    scaler = StandardScaler()
    for feature in features_to_scale:
        df[feature] = scaler.fit_transform(df[[feature]])
    return df

### -------------MISSINGS------------- ###

# Completa los missings del dataset (deben ser numéricos) usando Iterative Imputer (
# una implementación aproximada de MICE)
def fill_with_iterative_imputer(X):
    imputer = IterativeImputer(estimator=BayesianRidge(), 
                            n_nearest_features=None, # usamos todos los features para estimar
                            imputation_order='ascending', # empezamos rellenando los features con menor cantidad de missings
                            random_state=RANDOM_STATE,
                            initial_strategy='median',
                            verbose=2,
                            max_iter=4)

    imputer.fit(X)
    Xtrans = imputer.transform(X)
    df_imputed = pd.DataFrame(Xtrans, columns=X.columns)

    df_imputed = convert_features(df_imputed, df_imputed.columns, "float32")

    return df_imputed

# Para que en variables CATEGÓRICAS (pasadas en una lista "features_list"), tomemos a los NaNs 
# como una categoría más. Inplace
def nan_as_a_class(df, features_list):
    for feature in features_list:
        df[feature].fillna('missing', inplace=True)

# Returns the datafram with the instances that have less than "nan_proportion" of NaNs
def remove_instances_with_nan_proportion(df, nan_proportion):
    return df[df.isnull().mean(axis=1) < nan_proportion]

# La strategy puede ser median, mean o most_feaquent. Devuelve el dataframe
def fill_nan_with_simple_imputer(df, features_to_fill, strategy):
    simple_imputer = SimpleImputer(strategy=strategy)
    for feature in features_to_fill:
        df[feature] = simple_imputer.fit_transform(df[[feature]])

    return df

# Sirve para convertir valores no posibles/erráticos (que no corresponden a la naturaleza de dicho feature)
# y reemplazarlos por NaN. Inplace
def convert_values_to_nan(df, feature, values_list):
    df[feature].replace(values_list, np.nan, inplace=True)


### --------------CATEGORIAS----------- ###

# Convierte los features pasados en la lista al tipo type, a fin de reducir 
# el espacio en memoria que ocupa el dataset
def convert_features(df,features_to_convert,type):
    for row in features_to_convert:
        df[row] = df[row].astype(type)

    return df
