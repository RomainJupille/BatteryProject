import pandas as pd
from BatteryProject.params import *


def get_data_gcp(FEATURE_NAME,NROWS = None, RANGE_COL = None):

    #FEATURE_NAME = f'{input()}.csv'
    #FEATURE_NAME = "Temporary_data/summary-charge-duration.csv"
    #a = int(input("Entrez le premier index de la colonne souhaité : "))
    #b = int(input("Entrez le dernier index de la colonne souhaité : "))

    df = pd.read_csv(f"gs://{BUCKET_NAME}/transformed_data/{FEATURE_NAME}.csv",nrows = NROWS, usecols=RANGE_COL)

    return df

def get_data_local(FEATURE_NAME,NROWS = None, RANGE_COL = None):

    df = pd.read_csv(f"../..//{FEATURE_NAME}.csv",nrows = NROWS, usecols=RANGE_COL)

    return df

if __name__ == '__main__':
    df = get_data_gcp("summary_charge_capacity",None, None)
    df.info()
