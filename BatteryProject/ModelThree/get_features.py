
import pandas as pd
import numpy as np

def get_features_target(df_dict, deep, offset,indexes):

    X = None
    y = []
    n_features = len(df_dict)
    for i in indexes:
        nb_nan = df_dict['disc_capa'].iloc[i,:].isna().sum()
        for j in range((3000 - nb_nan - deep)//offset):
            sample = np.zeros((deep,n_features))
            for k, df in enumerate(df_dict.values()):
                sample[:,k] = np.array(df.iloc[i,1+offset * j:1+offset*j + deep])
            if X is None:
                X = sample.reshape(1,deep,n_features)
            else:
                X = np.concatenate([X,sample.reshape(1,deep,n_features)], axis = 0)

            y.append(3000 - (offset*j + deep + nb_nan))

    y = np.array(y)

    return X, y
