import pandas as pd
from google.cloud import storage
import numpy as np
import pandas as pd
from BatteryProject.params import *


def get_data():
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=1000)
    return df

if __name__ == '__main__':
    df = get_data()
