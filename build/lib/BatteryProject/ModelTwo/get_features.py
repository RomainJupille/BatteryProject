

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences


def generate_fake_Xy(deep = 5):
    ''' génère des Xy bidons pour tester les shapes
        afin de fiter.
        il faudra ensuite convertir les vrais données
    '''
    disc_capa = np.repeat([1.068216], deep)
    dis_ener = np.repeat([6.191177], deep)
    temp_avg = np.repeat([29.922943], deep)
    char_capa = np.repeat([1.441996], deep)
    features = [
        disc_capa,
        dis_ener,
        temp_avg,
        char_capa
    ]
    X = np.array(features * 134)
    y = np.random.randint(low=300, high=1500, size=134)
    y = np.repeat(y,4)
    #y = np.reshape(y, (-1, 1))
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

    # target
    target = []
    for cap in df_dict['disc_capa'].iloc:
        distance_to_end = len(cap[:].dropna()) - (offset + deep)
        target.append(distance_to_end)
    target = pd.Series(target)


    ###########################################################
    # TODO:
    # 1 - ne pas utiliser un DataFrame!
    # 2 - il faut transposer les données (4 features au total)
    # 3 - il faut traiter les cas ou "offset+deep" récupèrent des NaN
    #     (il faudra les remplacer par des 0 et les masker durant le CNN)
    #
    #
    # données acutelles (avec deep=2):
    #   [a1,a2, b1,b2, c1,c2, d1,d2, ...]
    #
    # données attendues par le modèle (avec deep=2):
    #   np.array([[a1,a2, ...]
    #             [b1,b2, ...]
    #             [c1,c2, ...]
    #             [d1,d2, ...]])
    #
    ###########################################################
    # features
    features = pd.DataFrame()
    check = pd.DataFrame()
    for key, values in df_dict.items():
        #print(key, values)
        for column in values.iloc[:,offset+1:offset+deep+1].columns:
            features[f'{key}_{int(column) - offset}'] = df_dict[key][column]

    # TODO: TEST, à virer (quand le modèle sera capable de prédire)
    features,target = generate_fake_Xy(deep)

    return features, target
