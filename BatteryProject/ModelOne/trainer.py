from email.errors import NoBoundaryInMultipartDefect
from pickle import TRUE
from pyexpat import model
from google.cloud import storage
import numpy as np
import pandas as pd
from sklearn import metrics
import joblib
from termcolor import colored
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import mlflow
import csv
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import learning_curve

from BatteryProject.data import get_data_local, get_data_gcp
from BatteryProject.ModelOne.get_features import get_features_target
from BatteryProject.ModelOne.model_params import features, models, scalers
from BatteryProject.params import *




class Trainer():

    def __init__(self, features_name = None, deep = 5, classes = [550], grid_params ={} ):
        """
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
        for name, path in self.features_name.items():
            df = get_data_local(path)
            df_dict[name] = df

        self.raw_data = df_dict
        self.X, self.y = get_features_target(self.raw_data, deep = self.deep, classes = self.classes)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state = 0)

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
        self.scaler_name = str(self.pipeline["scaler"]).split('(', 1)[0]
        self.model_name = str(self.pipeline["model"]).split('(', 1)[0]

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
        dir_path = os.path.join(os.path.dirname(__file__),'Models','IDs.csv')
        df_ids = pd.read_csv(dir_path)
        last_id = df_ids.values.max()
        new_id = last_id + 1

        with open(dir_path, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([new_id.tolist()])
        if new_id < 9:
            self.ID = f"000{new_id}"
        elif new_id < 99:
            self.ID = f"00{new_id}"
        elif new_id < 999:
            self.ID = f"0{new_id}"
        else:
            self.ID = f"{new_id}"
        return self.ID


    def save_model(self):
        """Save the model into a .joblib format"""

        #get the ID
        self.create_save_id_model()

        #Save in MLFlow
        EXPERIMENT_NAME = f"[FR] [Marseille] [TomG13100] Battery_ModelOne + 1"
        self.experiment_name = EXPERIMENT_NAME
        self.mlflow_log_param("ID", self.ID)

        self.mlflow_log_param("model", self.model_name)
        self.mlflow_log_param("scaler", self.scaler_name)

        for feature in list(self.features_name.keys()):
            self.mlflow_log_param(feature, 1)

        for key, values in self.grid_search.best_params_.items():
            self.mlflow_log_param(key.split('__')[1], values)

        self.mlflow_log_metric("accuracy", self.evaluation['accuracy'])
        self.mlflow_log_metric("precision", self.evaluation['precision'])
        self.mlflow_log_metric("roc_auc", self.evaluation['roc_auc'])



        # save the model.joblib (locally)
        joblib.dump(self.grid_search.best_estimator_, f'BatteryProject/ModelOne/Models/model_{self.ID}.joblib')
        model_name = f"model_{self.ID}"


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
    def mlflow_log_params(self, dict_params):
        self.mlflow_client.log_params(self.mlflow_run.info.run_id, dict_params)
    def mlflow_log_metric(self,key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == '__main__':
    for feat in features.values():
        for param in models.values():
            for scal in scalers.values():
                mod = param[0]
                grid = param[1]
                trainer = Trainer()
                trainer.features = feat
                trainer.get_data(feat)
                trainer.set_pipeline(scaler = scal, model = mod)
                trainer.run(grid)
                trainer.eval()
                trainer.save_model()
