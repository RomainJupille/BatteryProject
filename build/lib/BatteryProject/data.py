import pandas as pd
import os

from BatteryProject.params import *


def get_data_gcp(FEATURE_NAME,NROWS = None, RANGE_COL = None):

    #FEATURE_NAME = f'{input()}.csv'
    #FEATURE_NAME = "Temporary_data/summary-charge-duration.csv"
    #a = int(input("Entrez le premier index de la colonne souhaité : "))
    #b = int(input("Entrez le dernier index de la colonne souhaité : "))

    df = pd.read_csv(f"gs://{BUCKET_NAME}/transformed_data/{FEATURE_NAME}.csv",nrows = NROWS, usecols=RANGE_COL)

    return df

def get_data_local(feature_name,nrows = None, range_col= None):
    dir_path = os.path.dirname(__file__)
    print(dir_path)
    transformed_data_path = os.path.join(dir_path, "..", "..", "raw_data", "transformed_data", feature_name)
    transformed_data_path = os.path.normpath(transformed_data_path)
    df = pd.read_csv(transformed_data_path,nrows = nrows, usecols=range_col)

    return df

if __name__ == '__main__':
    df = get_data_local("summary_charge_capacity.csv",None, None)
    df.info()
