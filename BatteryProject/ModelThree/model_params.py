from tensorflow.keras.layers import LSTM

features = [
    {
        'disc_capa' : 'summary_discharge_capacity.csv',
        'dis_ener' : 'summary_discharge_energy.csv',
        'char_capa' : 'summary_charge_capacity.csv'},
    {
        'disc_capa' : 'summary_discharge_capacity.csv'},
    #     'disc_capa' : 'summary_discharge_capacity.csv',
    #     'dis_ener' : 'summary_discharge_energy.csv',
    #     'eff_ener' : 'summary_energy_efficiency.csv',
    #     'char_capa' : 'summary_charge_capacity.csv',
    #     'char_ener' : 'summary_charge_energy.csv',
    #     'dc_res' : 'summary_dc_internal_resistance.csv'},
    {
        'disc_capa' : 'summary_discharge_capacity.csv',
        'dis_ener' : 'summary_discharge_energy.csv',
        'eff_ener' : 'summary_energy_efficiency.csv',
        'char_capa' : 'summary_charge_capacity.csv',
        'char_ener' : 'summary_charge_energy.csv',
        'dc_res' : 'summary_dc_internal_resistance.csv',
        'temp_avg' : 'summary_temperature_average.csv',
        'temp_min' : 'summary_temperature_minimum.csv',
        'temp_max' : 'summary_temperature_maximum.csv'
        }
    ]

deeps_offset = [
    # {
    #     'deep' : 10,
    #     'offset' : 15
    # },
    # {
    #     'deep' : 20,
    #     'offset' : 20
    # },
    {
        'deep' : 40,
        'offset' : 40
    }
]

unit_types = ['LSTM']
n_layers = [2,3]
n_units = [4,5]
dropout = [0.3]
dropout_layer = [True]
