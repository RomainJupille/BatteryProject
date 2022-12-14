import numpy as np

def get_features_target(df_dict, deep, offset,indexes):

    X = None
    y = []
    barcode = []
    n_features = len(df_dict)
    for i in indexes:
        nb_nan = df_dict['disc_capa'].iloc[i,:].isna().sum()
        for j in range((3000 - nb_nan - deep)//offset):
            sample = np.zeros((deep,n_features + 1))
            for k, df in enumerate(df_dict.values()):
                sample[:,k] = np.array(df.iloc[i,1+offset * j:1+offset*j + deep])
            sample[:,n_features] = np.arange(offset * j, offset*j + deep)
            if X is None:
                X = sample.reshape(1,deep,n_features +1)
            else:
                X = np.concatenate([X,sample.reshape(1,deep,n_features +1)], axis = 0)

            y.append(3000 - (offset*j + deep + nb_nan))
            barcode.append(df.iloc[i,0])

    barcode = np.array(barcode)
    y = np.array(y).astype('float32')

    return X, y, barcode
