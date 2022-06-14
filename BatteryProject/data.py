import pandas as pd
from BatteryProject.params import *


def get_data():
    print("Entrez le nom d'une feature(fichier)")

    FEATURE_NAME = f'{input()}.csv'
    #FEATURE_NAME = "Temporary_data/summary-charge-duration.csv"
    a = int(input("Entrez le premier index de la colonne souhaité : "))
    b = int(input("Entrez le dernier index de la colonne souhaité : "))
    NCOL = range(a, b)


    df = pd.read_csv(f"gs://{BUCKET_NAME}/Temporary_data/{FEATURE_NAME}", usecols=NCOL)

    return df


if __name__ == '__main__':
    df = get_data()
    df.info()
