from tensorflow.keras.layers import LSTM

features = [
    {'disc_capa' : 'summary_discharge_capacity.csv',
     'char_capa' : 'summary_charge_capacity.csv'}]

deeps_offset = [
    {
        'deep' : 10,
        'offset' : 10
    },
    {
        'deep' : 20,
        'offset' : 20
    },
    # {
    #     'deep' : 40,
    #     'offset' : 40
    # },
    #     {
    #     'deep' : 40,
    #     'offset' : 80
    # }
]

unit_types = ['LSTM']
n_layers = [1]
n_units = [4]
dropout = [0.2]
dropout_layer = [True]
