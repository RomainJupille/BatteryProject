
from sklearn.ensemble import RandomForestClassifier
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
                LogisticRegression(max_iter= 3000),
                [
                {
                'model__penalty' : ['none'],
                'model__solver' : ['lbfgs', 'newton-cg', 'saga']
                },
                {
                'model__penalty' : ['l2'],
                'model__solver' : ['lbfgs', 'sag'],
                'model__C' : [0.01, 0.02, 0.05,0.1,0.2, 0.5, 1.0, 2.0, 5.0]
                },
                {
                'model__penalty' : ['l2','l1'],
                'model__solver' : ['liblinear'],
                'model__C' : [0.01, 0.02, 0.05,0.1,0.2, 0.5, 1.0, 2.0, 5.0]
                },
                {
                'model__penalty' : ['elasticnet'],
                'model__solver' : ['saga'],
                'model__C' : [0.01, 0.02, 0.05,0.1,0.2, 0.5, 1.0, 2.0, 5.0],
                'model__l1_ratio' : [0.0, 0.2, 0.4,0.6,0.8, 1.0]
                }
                ]
            ],
    'random_forest' : [
                RandomForestClassifier(),
                [
                {
                'model__n_estimators' : [50,100,200,400,800,1600],
                'model__min_samples_split' : [1,2,4,6,8],
                'model__min_samples_leaf' : [1,2,3,4]
                }
                ]
            ]

        }
