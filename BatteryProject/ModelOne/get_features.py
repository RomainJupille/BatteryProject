
import pandas as pd

def get_features_target(df_dict, deep, classes):
    '''
    Get the features and target form the dataframes in the df_dict
    df_dict must be given as a dict of dataframe
    keys of df_dict are used to define the titles of the features
    'Deep' defines how many columns are put inside the feature
    (each column is a chare/discharge cycle)

    Transform the summary_discharge_capacity into the target vector
    Return a df encoded of len(classes) + 1 classes
    Classes must be a list of int, each int being a category
    '''
    #Check for all the features if there is a a NaN in the first columns
    i = 0
    for key, values in df_dict.items():
        if i == 0:
            filter = (df_dict[key].iloc[:,1:deep+1].isna().sum(axis = 1) > 0) * 1
        else:
            filter = filter + (df_dict[key].iloc[:,1:deep+1].isna().sum(axis = 1) > 0) * 1
        i+= 1

    #Filter the rows where NaN have been found
    for key, values in df_dict.items():
        df_dict[key] = df_dict[key][filter == 0]

    #Define the target
    target = df_dict['disc_capa'].iloc[:,classes[0]].isna() * (-1)
    n = len(classes)
    if n > 1:
        for i in range(1,len(classes)):
            target = target + df_dict['disc_capa'].iloc[:,classes[i]].isna() * (-1)
    target = target + n

    #Define the X vector for all the features selected
    features = pd.DataFrame()
    check = pd.DataFrame()
    for key, values in df_dict.items():
        if check.shape[0] != 0:
            assert(check.all() == values.iloc[:,0].all())
        check = values.iloc[:,0]
        for column in values.iloc[:,1:deep +1].columns:
            features[f'{key}_{column}'] = df_dict[key][column]

    return features, target
