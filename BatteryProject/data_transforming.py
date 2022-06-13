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
    transformed_data_path = os.path.join(dir_path, "..", "..", "raw_data", "transformed_data_test")
    transformed_data_path = os.path.normpath(transformed_data_path)

    # récupération des noms des fichiers
    onlyfiles = [f for f in os.listdir(initial_data_path) if f[-1] != 'r']

    # définition des données du 1er fichier de donnée
    first_file_path = os.path.join(initial_data_path, onlyfiles[0])
    df = pd.read_json(first_file_path)
    indexes_list = df.index
    column_list = df.columns

    for file_name in onlyfiles[0:5]:
        file_path = os.path.join(initial_data_path, file_name)
        df = pd.read_json(file_path)
        valide_shape(df,indexes_list,column_list,one_val_cols_list,NaN_cols_list)
        add_lines(df,transformed_data_path)

def valide_shape(df,indexes,cols,one_val_cols,NaN_cols):
    assert(df.index.all() == indexes.all())
    assert(df.columns.all() == cols.all())
    for col in one_val_cols:
        assert(len(set(df[col].values))== 1)
    for col in NaN_cols:
        assert(df[col].isna().sum()== 21)

def initialize_files(path):

    pass

def add_lines(df,path):
    #write on the main file listing batteries
    file_path = os.path.join(path, 'test_details.csv')

    dict = {}
    for val in one_val_cols_list:
        dict[val] = df[val].iloc[0]
    with open(file_path, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict.keys())
        writer.writeheader()
        writer.writerow(dict)
    print("file written")


feature_list = [
    ['cycle_index', 'summary'],

    ['discharge_capacity', 'summary'],
    ['charge_capacity', 'summary'],

    ['discharge_energy', 'summary'],
    ['charge_energy', 'summary'],

    ['dc_internal_resistance', 'summary'],
    ['energy_efficiency', 'summary'],
    ['charge_throughput', 'summary'],
    ['energy_throughput', 'summary'],
    ['charge_duration', 'summary'],


    ['temperature_maximum', 'summary'],
    ['temperature_average', 'summary'],
    ['temperature_min', 'summary']]


transform_data()
