

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences


# for testing purposes
def generate_fake_Xy(deep = 5):
    features = [[1.068216, 6.191177, 29.922943, 1.441996]]
    X = [features * deep]
    X = np.array(X * 134)
    y = np.random.randint(low=300, high=1500, size=134)
    return X,y


def get_features_target(df_dict, offset, deep, classes):
    '''
    Get the features and target form the dataframes in the dict
    df_dict must be given as a dict of dataframe
    keys of df_dict are used to define the titles
    deep defines how many columns are put inside the feature
    (each cols being a chare/discharge cycle)
    '''
    #transform dataframe to drop columns with nan

    i = 0
    for key, values in df_dict.items():
        if i == 0:
            filter = (df_dict[key].iloc[:,1:deep+1].isna().sum(axis = 1) > 0) * 1
        else:
            filter = filter + (df_dict[key].iloc[:,1:deep+1].isna().sum(axis = 1) > 0) * 1
        i+= 1

    for key, values in df_dict.items():
        df_dict[key] = df_dict[key][filter == 0]


    # generation de la target
    target = []
    for cap in df_dict['disc_capa'].iloc:
        distance_to_end = len(cap[:].dropna()) - (offset + deep)
        target.append(distance_to_end)
    target = pd.Series(target)

    # generation features dans une windows (offset + deep)
    features = pd.DataFrame()
    check = pd.DataFrame()
    for key, values in df_dict.items():
        for column in values.iloc[:,offset+1:offset+deep+1].columns:
            index = int(column) - offset
            features[f'{key}_{index}'] = df_dict[key][column]

    # conversion dataframe -> numpy.array
    np_features = []
    for i in range(134):
        dim = []
        for j in range(deep):
            dim.append(*features.iloc[i:i+1,j::deep].values.tolist())
        np_features.append(dim)
    np_features = np.array(np_features)

    return np_features, target
