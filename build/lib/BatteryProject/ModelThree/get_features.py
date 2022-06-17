
import pandas as pd
import numpy as np

def get_features_target(df_dict, deep, offset,indexes):

    X = None
    y = []

    for i in indexes:
        nb_nan = df_dict['disc_capa'].iloc[i,:].isna().sum()
        for j in range((3000 - nb_nan - deep)//offset):
            sample = np.zeros((deep,len(df_dict)))
            for k, df in enumerate(df_dict.values()):
                sample[:,k] = np.array(df.iloc[i,offset * j:offset*j + deep])
        if X is None:
            X = sample.reshape(1,20,2)
        else:
            X = np.concatenate([X,sample.reshape(1,20,2)], axis = 0)

        y.append(3000 - (offset*i + deep + nb_nan))

    return X, y
