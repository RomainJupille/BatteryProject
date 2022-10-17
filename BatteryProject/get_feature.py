import pandas as pd
import os
import BatteryProject.transform_params as tp

def get_data_local(feature_name,nrows = None, range_col= None):
    """
    Return a dict of df, each df correspond to a feature
    nrows and range_col respectively limit the number of row / columns returned by the method
    """
    dir_path = os.path.dirname(__file__)
    transformed_data_path = os.path.join(dir_path, "..", "..", tp.transformed_data_folder_name, feature_name)
    transformed_data_path = os.path.normpath(transformed_data_path)
    df = pd.read_csv(transformed_data_path,nrows = nrows, usecols=range_col)

    return df

if __name__ == '__main__':
    df = get_data_local("summary_charge_capacity.csv",None, None)
    df.info()
