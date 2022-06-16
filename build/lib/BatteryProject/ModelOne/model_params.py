
from sklearn.linear_model import LogisticRegression

features = {
    'feature_one' : {
        'disc_capa' : 'summary_discharge_capacity.csv',
        'dis_ener' : 'summary_discharge_energy.csv',
        'temp_avg' : 'summary_temperature_average.csv',
        'char_capa' : 'summary_charge_capacity.csv'}
}


models = {
    'log' : [   LogisticRegression(),
                [
                    {
                    'Logistic__penalty' : ['l2', None],
                    'Logistic__solver' : ['lbfgs', 'sag'],
                    'Logistic_C' : [0.1, 0.5, 1.0, 5.0, 10.0]
                    },
                    {
                    'Logistic__penalty' : ['l2', 'l1'],
                    'Logistic__solver' : ['liblinear'],
                    'Logistic_C' : [0.1, 0.5, 1.0, 5.0, 10.0]
                    }
                ]
            ]


        }
