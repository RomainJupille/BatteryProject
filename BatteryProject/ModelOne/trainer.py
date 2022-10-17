from pickle import TRUE
from unittest import expectedFailure
import numpy as np
import pandas as pd
import joblib
#from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
#import mlflow
import csv
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import learning_curve

from BatteryProject.data import get_data_local
from BatteryProject.ModelOne.get_features import get_features_target
from BatteryProject.ModelOne.model_params import features, models, model_scalers
# from BatteryProject.params import *


"""
The 'Model One' consist in predicting whether a battery will support more than a given number of cycles

It can be a binary classification (with only one value, the used value is 550 cycles)

It can also be used to differenciate several classes
"""


class Trainer():
    """
    Trainer is a class containing all the information for traning and optimize a model
    One trainer instance has given classes and given features
    """

    def __init__(self, features_name = None, deep = 5, classes = [550], grid_params ={} ):
        """
        Deep : number of cycle used for prediction
        Classes : [int, ...]. list of treshold used to create the different target classes of the model
        features : list of features (in the shape of a dictionnary)
        """
        self.features_name = features_name
        self.deep = deep
        self.classes = classes

        if len(classes) > 1 :
            self.binary = False
        else :
            self.binary = TRUE

        self.target_name = 'disc_capa'
        self.grid_params = grid_params

        # for MLFlow
        self.experiment_name = None

    def get_data(self, features_name):
        '''
        Get X and y from the list of features in feature_name
        Nan cleaning (done bu get_feature_target)
        Train_test_split
        '''
        self.features_name = features_name

        df_dict = {}

        #Get the data
        for name, path in self.features_name.items():
            df = get_data_local(path)
            df_dict[name] = df

        self.raw_data = df_dict

        self.X, self.y = get_features_target(self.raw_data, deep = self.deep, classes = self.classes)
        self.X_train, self.X_test, self.y_train, self.y_test, self.raw_train, self.raw_test = train_test_split(self.X, self.y, self.raw_data[self.target_name], test_size = 0.3, random_state = 0)

        return self

    def set_pipeline(self, scaler = RobustScaler(), model = LogisticRegression(max_iter = 3000)):
        self.scaler = scaler
        self.model = model

        pipe = Pipeline([
            ('scaler', self.scaler),
            ('model', self.model)])

        self.pipeline = pipe

        return self

    def run(self, grid_params):
        """ Run a Grid Search on the grid search params """
        self.grid_params = grid_params
        gs_results = GridSearchCV(self.pipeline, self.grid_params, n_jobs = -1, cv = 5, scoring="accuracy", verbose = 0)
        self.scaler_name = str(self.pipeline["scaler"]).split('(', 1)[0]
        self.model_name = str(self.pipeline["model"]).split('(', 1)[0]

        gs_results.fit(self.X_train, self.y_train)

        self.grid_search = gs_results

        return self

    def print_learning_curve(self):

        self.best_model = self.grid_search.best_estimator_["model"]
        train_sizes = np.linspace(0.1,1.0,10)
        train_sizes, train_scores, test_scores = learning_curve(
        self.best_model, X=self.X, y=self.y, train_sizes=train_sizes, cv=5, n_jobs=-1)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.figure(figsize=(15,7))
        plt.plot(train_sizes, train_scores_mean, label = 'Training score')
        plt.plot(train_sizes, test_scores_mean, label = 'Test score')
        plt.ylabel('accuracy score', fontsize = 14)
        plt.xlabel('Training set size', fontsize = 14)
        plt.title('Learning curves', fontsize = 18, y = 1.03)

        plt.legend()

    def eval(self):
        prediction = self.grid_search.best_estimator_.predict(self.X_test)
        dic = {
        'accuracy' : accuracy_score(prediction, self.y_test),
        'precision' : precision_score(prediction, self.y_test),
        'roc_auc' : roc_auc_score(prediction, self.y_test)
        }
        self.evaluation = dic
        return self.evaluation

    def save_model(self):
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Models','Models_records.csv')

        try:
            df_records = pd.read_csv(dir_path)

            try:
                df_records = df_records.drop(columns='Unnamed: 0')
            except:
                pass

            if df_records.shape[0] == 0:
                print("1st row added to the model")
                print('\n')
                new_id=1
            else:
                last_id = df_records['Try_ID'].values.max()
                new_id = last_id + 1
                print("adding a row in the model")
                print('\n')

        except:
            df_records = pd.DataFrame()
            print("creating the dataframe")
            print('\n')
            new_id=1



        data = pd.DataFrame()
        data['Try_ID'] = [new_id]
        data['Model'] = [self.model_name]
        data['Scaler'] = [self.scaler_name]

        for key in self.features_name.keys():
            data[f"Features_{key}"] = ['X']

        for key, value in self.evaluation.items():
            data[f"Metrics_{key}"] = [value]

        for key, values in self.grid_search.best_params_.items():
            data[f"HyperParams_{key.split('__')[1]}"] = [values]
        print("==== data added to the df======")
        print(data)
        print('\n')
        df_records = df_records.append(data, ignore_index=True)

        col_list = df_records.columns
        col_l1 = ['Try_ID',  'Model', 'Scaler']
        col_l2 = [a for a in col_list if a[0:8] == 'Features']
        col_l3 = [a for a in col_list if a[0:7] == 'Metrics']
        col_l4 = [a for a in col_list if a[0:11] == 'HyperParams']
        new_col_list = col_l1 + col_l2 + col_l3 + col_l4
        df_records = df_records[new_col_list]

        print("saving the record")
        print(df_records)
        print("\n")
        df_records.to_csv(dir_path)

        if new_id <= 9:
            self.ID = f"000{new_id}"
        elif new_id < 99:
            self.ID = f"00{new_id}"
        elif new_id < 999:
            self.ID = f"0{new_id}"
        else:
            self.ID = f"{new_id}"

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Models', f"model_{self.ID}.joblib")
        joblib.dump(self.grid_search.best_estimator_, model_path)

    def save_data(self):
        raw_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"raw_data_{self.ID}.csv")
        X_test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"X_test_{self.ID}.csv")
        y_test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"y_test_{self.ID}.csv")

        self.raw_test.to_csv(raw_data_path)
        np.savetxt(X_test_path , self.X_test, delimiter=",")
        np.savetxt(y_test_path, self.y_test, delimiter=",")

if __name__ == '__main__':
    #trainer = Trainer()
    #feat = features['feature_four']
    #trainer.get_data(feat)
    #trainer.save_test_csv()

    n_feat = len(features.values())
    n_param = len(models.values())
    n_scalers = len(model_scalers.values())
    print(f"Testing {n_feat * n_param * n_scalers} combinations")
    i = 1
    for feat in features.values():
       for param in models.values():
           for scal in model_scalers.values():
              mod = param[0]
              grid = param[1]
              trainer = Trainer()
              trainer.get_data(feat)
              trainer.set_pipeline(scaler = scal, model = mod)
              trainer.run(grid)
              trainer.eval()
              trainer.save_model()
              trainer.save_data()

              print(f"==== {i} combination ====")
              print(list(feat.keys()))
              print(trainer.grid_search.best_estimator_)
              print(trainer.grid_search.best_params_)
              print(trainer.evaluation)
              print(trainer.ID)
              print('\n')

              i+= 1
