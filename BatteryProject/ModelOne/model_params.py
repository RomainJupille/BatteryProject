
from sklearn.linear_model import LogisticRegression

features = {
    'feature_one' : {
        'disc_capa' : 'summary_discharge_capacity.csv',
        'dis_ener' : 'summary_discharge_energy.csv',
        'temp_avg' : 'summary_temperature_average.csv',
        'char_capa' : 'summary_charge_capacity.csv'}
}


models = {
    'log' : [
                LogisticRegression(),
                [
                {
                'Model__penalty' : ['none'],
                'Model__solver' : ['liblinear', 'newton-cg', 'saga']
                },
                {
                'Model__penalty' : ['l2'],
                'Model__solver' : ['lbfgs', 'sag'],
                'Model__C' : [0.01, 0.02, 0.05,0.1,0.2, 0.5, 1.0, 2.0, 5.0]
                },
                {
                'Model__penalty' : ['l2','l1'],
                'Model__solver' : ['liblinear'],
                'Model__C' : [0.01, 0.02, 0.05,0.1,0.2, 0.5, 1.0, 2.0, 5.0]
                },
                {
                'Model__penalty' : ['elasticnet'],
                'Model__solver' : ['saga'],
                'Model__C' : [0.01, 0.02, 0.05,0.1,0.2, 0.5, 1.0, 2.0, 5.0],
                'Model__l1_rate' : [0.0, 0.2, 0.4,0.6,0.8, 1.0]
                }
                ]
            ]
        }
