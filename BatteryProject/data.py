import pandas as pd
from BatteryProject.params import *


def get_data():
    print("Entrez le nom d'une feature(fichier)")
    FEATURE_NAME = f'transformed_data/{input()}.csv'
    a = input("Entrez le premier index de la colonne souhaité")
    b = input("Entrez le dernier index de la colonne souhaité")
    NCOL = range(a, b)
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{FEATURE_NAME}", usecols=NCOL)
    return df


if __name__ == '__main__':
    df = get_data()
