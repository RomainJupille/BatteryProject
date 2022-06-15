
import pandas as pd

def get_target(df, classes = [550]):
    '''
    transform the summary_discharge_capacity_into the target vector
    Return a df encoded of len(classes) + 1 classes
    Classes must be a list of int
    '''

    target = df.iloc[:,classes[0]].isna() * (-1)
    n = len(classes)
    if n > 1:
        for i in range(1,len(classes)):
            target = target + df.iloc[:,classes[i]].isna() * (-1)
    target = target + n

    return target

def get_features(df_dict, deep = 5):
    '''
    Get the features from the df
    df_dict must be given as a dict of dataframe
    keys of df_dict are used to define the titles
    deep defines how many columns are put inside the feature
    (each cols being a chare/discharge cycle)
    '''
    features = pd.DataFrame()
    check = pd.DataFrame()
    for key, values in df_dict.items():
        if check.shape[0] != 0:
            assert(check.all() == values.iloc[:,0].all())
        check = values.iloc[:,0]
        for column in values.iloc[:,1:6].columns:
            features[f'{key}_{column}'] = df_dict[key][column]

    return features
