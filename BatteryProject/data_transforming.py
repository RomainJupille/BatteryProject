import numpy as np
import os
import pandas as pd
import seaborn as sns
import csv

import matplotlib.pyplot as plt
from sqlalchemy import column, values


one_val_cols_list = ['@module', '@class', 'barcode', 'protocol', 'channel_id', '@version']
NaN_cols_list = ['diagnostic_summary', 'diagnostic_interpolated']

def transform_data():
    #récupération des paths vers les dossiers de données
    dir_path = os.path.dirname(__file__)
    initial_data_path = os.path.join(dir_path, "..", "..", "raw_data", "initial_data")
    initial_data_path = os.path.normpath(initial_data_path)
    transformed_data_path = os.path.join(dir_path, "..", "..", "raw_data", "transformed_data")
    transformed_data_path = os.path.normpath(transformed_data_path)

    # récupération des noms des fichiers
    onlyfiles = [f for f in os.listdir(initial_data_path) if f[-1] != 'r']

    # définition des données du 1er fichier de donnée
    first_file_path = os.path.join(initial_data_path, onlyfiles[0])
    df = pd.read_json(first_file_path)
    indexes_list = df.index
    column_list = df.columns

    initialize_files(df, transformed_data_path)
    i = 0
    for file_name in onlyfiles:
        file_path = os.path.join(initial_data_path, file_name)
        df = pd.read_json(file_path)
        valide_shape(df,indexes_list,column_list,one_val_cols_list,NaN_cols_list)
        add_lines(df,transformed_data_path)
        print(f"file {i}")
        i += 1

def valide_shape(df,indexes,cols,one_val_cols,NaN_cols):
    assert(df.index.all() == indexes.all())
    assert(df.columns.all() == cols.all())
    for col in one_val_cols:
        assert(len(set(df[col].values))== 1)
    for col in NaN_cols:
        assert(df[col].isna().sum()== 21)

def initialize_files(df, path):
    file_path = os.path.join(path, 'test_details.csv')
    with open(file_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=one_val_cols_list)
        writer.writeheader()
    for values in df.index:
        file_path = os.path.join(path, f"summary_{values}.csv")
        with open(file_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['barcode'] +[i for i in range(0,3000)])
            writer.writeheader()
    for values in df.index:
        file_path = os.path.join(path, f"cycles_interpolated_{values}.csv")
        with open(file_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['barcode'] +[i for i in range(0,3000)])
            writer.writeheader()

def add_lines(df,path):
    #write on the main file listing batteries
    file_path = os.path.join(path, 'test_details.csv')

    dict = {}
    for val in one_val_cols_list:
        dict[val] = df[val].iloc[0]
    with open(file_path, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict.keys())
        writer.writerow(dict)

    for values in df.index:
        if isinstance(df['summary'][values], float) == False:
            file_path = os.path.join(path, f"summary_{values}.csv")
            with open(file_path, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([df['barcode'].iloc[0]] + df['summary'][values])

    for values in df.index:
        if isinstance(df['cycles_interpolated'][values], float) == False:
            file_path = os.path.join(path, f"cycles_interpolated_{values}.csv")
            with open(file_path, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([df['barcode'].iloc[0]] + df['cycles_interpolated'][values])

    print("file written")


transform_data()
