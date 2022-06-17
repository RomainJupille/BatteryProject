from pyexpat import model
from google.cloud import storage
import numpy as np
import pandas as pd
from sklearn import metrics
import joblib
from BatteryProject.params import *
from termcolor import colored
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import mlflow
import csv
import os

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from BatteryProject.data import get_data_local, get_data_gcp
from BatteryProject.ModelOne.get_features import get_features_target
from BatteryProject.ModelOne.model_params import features
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


class Trainer():

    def __init__(self, features_name = None, deep = 5, classes = [550], grid_params ={} ):
        """
            features : list of features (in the shape of a dictionnary)
        """
        self.features_name = features_name
        self.deep = deep
        self.classes = classes
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
        for name, path in self.features_name.items():
            df = get_data_local(path)
            df_dict[name] = df

        self.raw_data = df_dict
        self.X, self.y = get_features_target(self.raw_data, deep = self.deep, classes = self.classes)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.3)

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
        gs_results = GridSearchCV(self.pipeline, self.grid_params, n_jobs = -1, cv = 5, scoring="accuracy")
        self.mlflow_log_param("model", "Linear")
        gs_results.fit(self.X_train, self.y_train)

        self.grid_search = gs_results

        return self

    def print_learning_curve(self):

        self.best_model = self.grid_search.best_estimator_["model"]
        train_sizes = list(range(20,108,10))
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

    def create_save_id_model(self):

        dir_path = os.path.dirname(__file__)
        if os.stat(os.path.join(dir_path, "nom du dossier", 'IDs.csv')).st_size == 0:
            nb = 1
            ID = f"00000{nb}"
        else:
            nb = 2
            ID = f"00000{nb}"

            if nb > 9:
                ID = f"0000{nb}"
            elif nb > 99:
                ID = f"000{nb}"
            elif nb > 999:
                ID = f"000{nb}"
            elif nb > 9999:
                ID = f"00{nb}"
            elif nb > 99999:
                ID = f"0{nb}"
            else:
                ID = f"00000{nb}"
        with open(f'IDs.csv', 'w', newline="") as csvfile:
            csv.writer(csvfile).writerow[ID]
        nb += 1


    def save_model(self, model):
        """Save the model into a .joblib format"""
        ids_df = pd.read_csv("IDs.csv")
        id = ids_df.tail(1)
        joblib.dump(model, f'Models/model_{id}.joblib')
        model_name = f"model_{id}"
        EXPERIMENT_NAME = f"[FR] [Marseille] [TomG13100] {model_name} + 1"
        self.experiment_name = EXPERIMENT_NAME
        return self.experiment_name

    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self,key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)


    def mlflow_log_metric(self,key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == '__main__':
    trainer = Trainer()
