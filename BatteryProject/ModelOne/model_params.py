
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

features = {
    'feature_one' : {
        'disc_capa' : 'summary_discharge_capacity.csv',
        'dis_ener' : 'summary_discharge_energy.csv',
        'temp_avg' : 'summary_temperature_average.csv',
        'char_capa' : 'summary_charge_capacity.csv'},

    'feature_two' : {
        'disc_capa' : 'summary_discharge_capacity.csv'},

    'feature_three' : {
        'disc_capa' : 'summary_discharge_capacity.csv',
        'dis_ener' : 'summary_discharge_energy.csv',
        'eff_ener' : 'summary_energy_efficiency.csv',
        'char_capa' : 'summary_charge_capacity.csv',
        'char_ener' : 'summary_charge_energy.csv',
        'dc_res' : 'summary_dc_internal_resistance.csv'},

    'feature_four' : {
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
}

scalers = {
        "RobustScaler" : RobustScaler(),
        "StandardScaler" : StandardScaler(),
        "MinMaxScaler" : MinMaxScaler()
        }

models = {
    'log' : [
                LogisticRegression(max_iter= 3000),
                [
                {
                'model__penalty' : ['none'],
                'model__solver' : ['lbfgs', 'newton-cg', 'saga'],
                'model__max_iter' : [500, 1000, 3000, 6000, 10000]
                },
                {
                'model__penalty' : ['l2'],
                'model__solver' : ['lbfgs', 'sag'],
                'model__C' : [0.005, 0.01, 0.02, 0.05,0.08, 0.1,0.015, 0.2, 0.5, 1.0, 2.0, 5.0],
                'model__max_iter' : [500, 1000, 3000, 6000, 10000]
                },
                {
                'model__penalty' : ['l2','l1'],
                'model__solver' : ['liblinear'],
                'model__C' : [0.005, 0.01, 0.02, 0.05,0.08, 0.1,0.015, 0.2, 0.5, 1.0, 2.0, 5.0],
                'model__max_iter' : [500, 1000, 3000, 6000, 10000]
                },
                {
                'model__penalty' : ['elasticnet'],
                'model__solver' : ['saga'],
                'model__C' : [0.005, 0.01, 0.02, 0.05,0.08, 0.1,0.15, 0.2, 0.5, 1.0, 2.0, 5.0],
                'model__l1_ratio' : [0.0, 0.2, 0.4,0.6,0.8, 1.0],
                'model__max_iter' : [500, 1000, 3000, 6000, 10000]
                }
                ]
            ],
    'random_forest' : [
                RandomForestClassifier(),
                [
                {
                'model__n_estimators' : [50,100,200,400],
                'model__min_samples_split' : [2,4,6],
                'model__min_samples_leaf' : [1,2,3,4]
                }
                ]
            ]

        }
