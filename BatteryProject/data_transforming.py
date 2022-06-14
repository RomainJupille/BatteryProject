import numpy as np
import os
import pandas as pd
import csv

one_val_cols_list = ['@module', '@class', 'barcode', 'protocol', 'channel_id', '@version']
NaN_cols_list = ['diagnostic_summary', 'diagnostic_interpolated']

#list of barcode to drop (this information is given by the paper)
barcode_to_drop = []

"""
POUR COMMENTAIRE :
mauvaises cellules batch 1
['EL150800465027',1]
['EL150800464002',1]
['EL150800463980',1]
['EL150800463882',1]
['EL150800460653',1]
mauvaises cellules batch 2
['el150800460596',2]
['el150800460518',2]
['el150800460605',2]
['el150800460451',2]
['el150800460478',2]
mauvaises cellules batch 3
['el150800737234',3]
['EL150800737380',3]
['el150800737386',3]
['el150800737299',3]
['el150800737350',3]
['el150800739477',3]
"""


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
    file_droped = 0
    file_concat = 0
    # files names to transform
    onlyfiles = [f for f in os.listdir(initial_data_path) if f[-1] != 'r']
    files_dict = {}

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

        bc = df['barcode'].iloc[0]

        if bc.upper() in barcode_to_drop:
            file_droped += 1
            print(f"{bc} has been dropped")
        else:
            add_lines_details(df,transformed_data_path)
            if bc in files_dict.keys():
                files_dict[bc].append(file_name)
                file_concat += 1
            else:
                files_dict[bc] = [file_name]
                i += 1

        print(f"file {i} read a 1st time and added to the dict")

    print(f"{i} files have been added in the dictionnary")
    print(file_droped)
    print(file_concat)
    print(len(files_dict))
    print(files_dict)

    i=0
    for barcode, file_names in files_dict.items():
        add_lines_data(barcode, file_names,initial_data_path,transformed_data_path)
        print(f"Barcodes {i} read and the data has been added to the csv files")
        i+=1

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
            writer = csv.DictWriter(csvfile, fieldnames=['barcode'] +[i for i in range(0,2_000_000)])
            writer.writeheader()

def add_lines_details(df,path):
    '''add a line on the csv file_details for each data file and save the barcodes / file names relations in a dict'''
    file_path = os.path.join(path, 'test_details.csv')

    dict = {}
    for val in one_val_cols_list:
        dict[val] = df[val].iloc[0]

    with open(file_path, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict.keys())
        writer.writerow(dict)

def add_lines_data(barcode, file_names,path_input,path_output):
    '''
    Add data to all csv files
    The method manages the duplicate codebars
    '''

    dict_df = {}

    for file_name in file_names:
        #create a dict containing all the df corresponding to barcode
        print(file_name)

        file_path = os.path.join(path_input, file_name)
        dict_df[file_name] = pd.read_json(file_path)

    values = dict_df[file_names[0]].index

    for value in values:
        data_list = []
        for key, df in dict_df.items():
            if isinstance(df['summary'][value], float) == False:
                data_list = df['summary'][value] + data_list
        data_list = [barcode] + data_list

        file_path = os.path.join(path_output, f"summary_{value}.csv")
        with open(file_path, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data_list)

    for value in values:
        data_list = []
        for key, df in dict_df.items():
            if isinstance(df['cycles_interpolated'][value], float) == False:
                nl =  np.array(df['cycles_interpolated'][value])
                try:
                    nl = nl.astype('float32')
                    print('reduction done')
                except:
                    pass
                nl = list(nl)
                data_list = nl + data_list
        data_list = [barcode] + data_list

        file_path = os.path.join(path_output, f"cycles_interpolated_{value}.csv")
        with open(file_path, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data_list)

    print("file written")

transform_data()







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
    """
    # utilisation:
    file_path = os.path.join(path, 'test_details.csv')
    file_path_out = os.path.join(path, 'test_details_out.csv')
    extract_protocol_file(file_path, file_path_out)
    """

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
    tmp.drop(columns=['@module'], inplace=True)
    tmp.drop(columns=['@class'], inplace=True)

    """ export """
    tmp.to_csv(csv_file_out, index=False)


def get_bad_cells_barcode(df):
    """
    # utilisation
    cells_to_drop = get_bad_cells_barcode(df)

    # observations à enlever (batch1)
    for c in cells_to_drop[0]:
        print(df[df["barcode"] == c])
    # observations à enlever (batch2)
    for c in cells_to_drop[1]:
        print(df[df["barcode"] == c])
    # observations à enlever (batch3)
    for c in cells_to_drop[2]:
        print(df[df["barcode"] == c])
    """
    bad_cells_b1 = [8, 10, 12, 13, 22]
    bad_cells_b2 = [7, 8, 9, 15, 16]
    bad_cells_b3 = [37, 2, 23, 32, 42, 43]
    batches_date = ['2017-05-12', '2017-06-30', '2018-04-12']
    bad_cells_batches = [ [], [], [] ]
    # batch 1
    tmp = df.copy()
    tmp = tmp.drop(tmp[tmp.batch == batches_date[1]].index)
    tmp = tmp.drop(tmp[tmp.batch == batches_date[2]].index)
    for bc in bad_cells_b1:
        bad_cells_batches[0].append(tmp[tmp['channel_id'] == bc].barcode.iloc[0])
    # batch 2
    tmp = df.copy()
    tmp = tmp.drop(tmp[tmp.batch == batches_date[0]].index)
    tmp = tmp.drop(tmp[tmp.batch == batches_date[2]].index)
    for bc in bad_cells_b2:
        bad_cells_batches[1].append(tmp[tmp['channel_id'] == bc].barcode.iloc[0])
    # batch 2
    tmp = df.copy()
    tmp = tmp.drop(tmp[tmp.batch == batches_date[0]].index)
    tmp = tmp.drop(tmp[tmp.batch == batches_date[1]].index)
    for bc in bad_cells_b3:
        bad_cells_batches[2].append(tmp[tmp['channel_id'] == bc].barcode.iloc[0])
    return bad_cells_batches

def preprocessing_batch_feature_to_numeric(df):
    batches_date = ['2017-05-12', '2017-06-30', '2018-04-12']
    for idx, batch_date in enumerate(batches_date):
        df.loc[df['batch'] == batch_date, "batch"] = idx
    return df
