#params, if require to change folder names
initial_data_folder_name = 'InitialData'
transformed_data_folder_name = 'TransformedData'
get_details = True
get_summary = True
get_cycles_interpolated = False

#list of barcode to drop (this information is given by the paper)
bc_to_drop = [
'EL150800465027',
'EL150800464002',
'EL150800463980',
'EL150800463882',
'EL150800460653',
'EL150800460596',
'EL150800460518',
'EL150800460605',
'EL150800460451',
'EL150800460478',
'EL150800737234',
'EL150800737380',
'EL150800737386',
'EL150800737299',
'EL150800737350',
'EL150800739477']

#list of unique values (added into the test_details file)
one_val_cols_list = ['@module', '@class', 'barcode', 'protocol', 'channel_id', '@version']

#columns not used during sampling
NaN_cols_list = ['diagnostic_summary', 'diagnostic_interpolated']
