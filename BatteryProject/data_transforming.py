import os
import pandas as pd
import csv

one_val_cols_list = ['@module', '@class', 'barcode', 'protocol', 'channel_id', '@version']
NaN_cols_list = ['diagnostic_summary', 'diagnostic_interpolated']

def transform_data():
    '''
    Transformation of the JSON files (one file per battery) into
    csv with concatenate data for a given measurement
    '''

    #path of the files when doing the transformation
    dir_path = os.path.dirname(__file__)
    initial_data_path = os.path.join(dir_path, "..", "..", "raw_data", "initial_data")
    initial_data_path = os.path.normpath(initial_data_path)
    transformed_data_path = os.path.join(dir_path, "..", "..", "raw_data", "transformed_data")
    transformed_data_path = os.path.normpath(transformed_data_path)

    # files names to transform
    onlyfiles = [f for f in os.listdir(initial_data_path) if f[-1] != 'r']

    # definition of the shape of the 1st file
    first_file_path = os.path.join(initial_data_path, onlyfiles[0])
    df = pd.read_json(first_file_path)
    indexes_list = df.index
    column_list = df.columns

    initialize_files(df, transformed_data_path)
    i = 0

    for file_name in onlyfiles:
        '''
        reading each JSON file, and spread data into multiple csv files
        data dispatching is done by add_line method
        '''
        file_path = os.path.join(initial_data_path, file_name)
        df = pd.read_json(file_path)
        valide_shape(df,indexes_list,column_list,one_val_cols_list,NaN_cols_list)
        add_lines(df,transformed_data_path)
        print(f"file {i}")
        i += 1

def valide_shape(df,indexes,cols,one_val_cols,NaN_cols):
    '''validate that the file has the same shape as the 1st file opened'''
    assert(df.index.all() == indexes.all())
    assert(df.columns.all() == cols.all())
    for col in one_val_cols:
        assert(len(set(df[col].values))== 1)
    for col in NaN_cols:
        assert(df[col].isna().sum()== 21)

def initialize_files(df, path):
    '''creation of the csv files with headers'''
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
    '''add in each csv file, one line of data (one per battery)'''
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



def extract_protocol_string(protocol):
    """ protocol extraction string """
    res = {}
    protocol = protocol.lower()

    tmp = protocol.split("\\")
    tmp1 = (tmp[1].split("-")[1]).split("c")

    batch = tmp[0]
    c1 = tmp1[0].split("c")[0]
    per = protocol.split("per")[0]
    c2 = protocol.split("per_")[1].split("c")[0]

    res['batch'] = batch.split("_")[0]
    res["c1"] = float(c1.replace("_","."))
    res["per"] = int(per[-2:].replace("_", ""))
    res["c2"] = float(c2.replace("_","."))
    res['newstructure'] = int(protocol.find("newstructure") >= 0)
    return res

def extract_protocol_list(protocols, value):
    """ protocol extraction list """
    res = []
    for protocol in protocols:
        res.append(extract_protocol_string(protocol)[value])
    return res

def extract_protocol_file(csv_file_in, csv_file_out):
    """ csv import """
    df = pd.read_csv(csv_file_in)
    tmp = df.copy()

    """ feature transformation """
    tmp['batch'] = extract_protocol_list(tmp['protocol'], "batch")
    tmp['c1'] = extract_protocol_list(tmp['protocol'], "c1")
    tmp['c2'] = extract_protocol_list(tmp['protocol'], "c2")
    tmp['per'] = extract_protocol_list(tmp['protocol'], "per")
    tmp['newstructure'] = extract_protocol_list(tmp['protocol'], "newstructure")

    """ clean """
    #tmp["barcode"] = tmp["barcode"].str.upper()

    """ drop """
    tmp.drop(columns=['protocol'], inplace=True)
    tmp.drop_duplicates(subset=['barcode'], inplace=True, ignore_index=True)
    #tmp.drop(columns=['@module'], inplace=True)
    #tmp.drop(columns=['@class'], inplace=True)

    """ export """
    tmp.to_csv(csv_file_out, index=False)

#file_path = os.path.join(path, 'test_details.csv')
#file_path_out = os.path.join(path, 'test_details_out.csv')
#extract_protocol_file(file_path, file_path_out)
