

import numpy as np
import pandas as pd


def get_features_target(df_dict, deep, offset,indexes):
    X = None
    y = []
    n_features = len(df_dict)

    for i in indexes:
        nb_nan = df_dict['disc_capa'].iloc[i,:].isna().sum()
        for j in range((3000 - nb_nan - deep)//offset):
            sample = np.zeros((deep,n_features + 1))
            for k, df in enumerate(df_dict.values()):
                sample[:,k] = np.array(df.iloc[i,1+offset * j:1+offset*j + deep])
            # ajoute une feature (l'offset)
            sample[:,n_features] = np.arange(offset * j, offset*j + deep)
            if X is None:
                X = sample.reshape(1,deep,n_features +1)
            else:
                X = np.concatenate([X,sample.reshape(1,deep,n_features +1)], axis = 0)

            y.append(3000 - (offset*j + deep + nb_nan))

    y = np.array(y)

    return X, y


"""def get_features_target_old(df_dict, offset, deep, idx):
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
    target = np.array(target)

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

    return np_features, target"""
