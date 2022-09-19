import numpy as np
import os
import pandas as pd
import csv
import transform_params as tp

def initialize_files(df, path, file_name = 'test_details.csv', details=True, summary=True, cycles=False):
    '''creation of the csv files with headers'''

    if details:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=tp.one_val_cols_list)
            writer.writeheader()
        print(f"{file_name} created")

    if summary:
        print("initializing summary files")
        for values in df.index:
            file_path = os.path.join(path, f"summary_{values}.csv")
            with open(file_path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['barcode'] +[i for i in range(0,3000)])
                writer.writeheader()
        print("summary_****.csv files initialized")

    if cycles:
        for values in df.index:
            file_path = os.path.join(path, f"cycles_interpolated_{values}.csv")
            with open(file_path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['barcode'] +[i for i in range(0,2_000_000)])
                writer.writeheader()

def valide_shape(df,indexes,cols,one_val_cols,NaN_cols):
    '''
    validate that i) each df has the same shape as the 1st df opened
    and that ii) each df has nan or unique values in specific columns
    '''
    #the new df has the same shape as the 1st one
    assert(df.index.all() == indexes.all())
    assert(df.columns.all() == cols.all())

    #unique values and nan columns are identified
    for col in one_val_cols:
        assert(len(set(df[col].values))== 1)
    for col in NaN_cols:
        assert(df[col].isna().sum()== 21)

    print('file shape is valid')

def add_lines_details(df,path, file_name = 'test_details.csv'):
    '''add a line on the csv file_details for each data file and save the barcodes / file names relations'''
    file_path = os.path.join(path, file_name)

    dict = {}
    #the only values added in this file are the one that are constant for a given battery
    #they are parameters of the model
    for val in tp.one_val_cols_list:
        dict[val] = df[val].iloc[0]

    with open(file_path, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict.keys())
        writer.writerow(dict)

def add_lines_data(barcode, file_names,path_input,path_output, summary=True, cycles=False):
    '''
    Add the data for a given battery (identified by its barcode) to the csv file
    File_names contain one or two names (some battery data are spread in two consecutive files)
    The method merge the data if there are two files
    '''

    dict_df = {}
    for file_name in file_names:
        #create a dict containing the df corresponding to barcode (one or two)

        file_path = os.path.join(path_input, file_name)
        dict_df[file_name] = pd.read_json(file_path)

    #creation of a variable 'values' containing all the measurement types
    values = dict_df[file_names[0]].index

    if summary:
        #for each measurement
        for value in values:
            #init an empty list
            data_list = []
            for df in dict_df.values():
                #if the data is not a 0.0 (empty)
                if isinstance(df['summary'][value], float) == False:
                    #the data is added to data list (one or two times)
                    data_list = df['summary'][value] + data_list
            #adding the barcode at the beginning of the file
            data_list = [barcode] + data_list

            #add a row in the csv file
            file_path = os.path.join(path_output, f"summary_{value}.csv")
            with open(file_path, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_list)

    #same method as above but for 'cycles' files
    #+ data compression to limit the size od the output csv files
    if cycles:
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

def extract_protocol_string(protocol):
    """ protocol extraction string """
    res = pd.DataFrame()
    #protocol = protocol.lower().encode('unicode_escape').decode()
    protocol = protocol.lower()
    tmp = protocol.split("\\")
    tmp1 = (tmp[1].split("-")[1]).split("c")

    batch = tmp[0]
    c1 = tmp1[0].split("c")[0]
    per = protocol.split("per")[0].split("_")[-1].split('-')[-1]
    c2 = protocol.split("per_")[1].split("c")[0]
    newstructure = int(protocol.find("newstructure") >= 0)

    return batch, c1, per, c2, newstructure

def clean_test_details(path, old_name = 'test_details.csv', new_name = 'test_details_clean.csv'):
    input_path = os.path.join(path, old_name)
    output_path = os.path.join(path, new_name)
    df = pd.read_csv(input_path)
    df['batch'], df['c1'], df['per'], df['c2'], df['newstructure'] = zip(*df['protocol'].map(extract_protocol_string))
    df = df.drop(columns = 'protocol')
    df.to_csv(output_path)
    pass


def transform_data(initial_data_folder_name,
                   transformed_data_folder_name,
                   details,
                   summary,
                   cylces,
                    barcode_to_drop):
    '''
    This method transform raw data from the paper (several JSON files, with one JSON corresponding to one battery)
    The output is one file correspond to one type of measurement and caontains the data of
    all the batteries for this given measurement
    The method also create a 'test_details file containing all the features of the batteries
    and important information regarding protocols
    The method can transform both 'summary' data and 'interpolated data'. However 'interpolated'
    are too large, therefore they are not used in this project
    '''
    #from a python file, automatically get the current directory
    #dir_path = os.path.dirname(__file__)

    #also possible to manualy give the current directory to the method
    dir_path = '/home/romainj/code/RomainJupille/wagon/Projet_batteries/BatteryProject/BatteryProject'


    #creation of paths towards initial folder and final folder
    initial_data_path = os.path.join(dir_path, "..", "..", initial_data_folder_name)
    initial_data_path = os.path.normpath(initial_data_path)
    transformed_data_path = os.path.join(dir_path, "..", "..", transformed_data_folder_name)
    transformed_data_path = os.path.normpath(transformed_data_path)

    # files names to transform (from initial folder)
    json_files = [f for f in os.listdir(initial_data_path) if f[-4:] == 'json']

    # definition of the shape of the 1st file
    first_file_path = os.path.join(initial_data_path, json_files[0])
    df = pd.read_json(first_file_path)
    #get indexes and columns of the data of the 1st battery
    indexes_list = df.index
    column_list = df.columns

    #initialization of the csv files in the output folder, based on columns and indexes of the 1st file
    #options
    initialize_files(df, transformed_data_path, details=details, summary=summary, cycles=cylces )


    '''
    for all the JSON files (one json corresponding to one battery)
    '''
    # values that will be dropped during processing
    file_droped = 0
    file_concat = 0
    files_dict = {}

    i = 0
    for file_name in json_files:
        #dowloading data into a dataframe
        file_path = os.path.join(initial_data_path, file_name)
        df = pd.read_json(file_path)

        #validation of the shape od the data that will be processed
        valide_shape(df,indexes_list,column_list,tp.one_val_cols_list,tp.NaN_cols_list)

        #get the barcode of the file
        bc = df['barcode'].iloc[0]

        #check if the data has to be dropped (some samples had issues during data acquisition)
        if bc.upper() in barcode_to_drop:
            file_droped += 1
            print(f"{bc} has been dropped")

        #if not add the file into the dict of bc (check for duplicated barcode)
        #One barcode can be linked to 1 or 2 files (some battery measurement extended over 2 periods)
        else:
            if bc in files_dict.keys():
                files_dict[bc].append(file_name)
                file_concat += 1
            else:
                files_dict[bc] = [file_name]
                add_lines_details(df,transformed_data_path)

        i += 1
        if i%10 ==0:
            print(f"{i} files checked ")

    clean_test_details(transformed_data_path)

    print('All files have been checked and the test_details file has been created')
    print(f"{file_droped} files droped")
    print(f"{file_concat} files concatenated")


    i=0
    #Finally, for each barcode saved previously, data are added into the csv files
    for barcode, file_names in files_dict.items():

        add_lines_data(barcode, file_names,initial_data_path,transformed_data_path, summary=summary  , cycles=cycles )
        print(f"Barcodes{i} read and the data has been added to the csv files")
        i+=1

    print(f"{i} lines created")

if __name__ == '__main__':
    transform_data(tp.initial_data_folder_name, tp.transformed_data_folder_name, tp.get_details, tp.get_summary, get_cycles_interpolated, bc_to_drop)
