from google.cloud import storage
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from BatteryProject.data import get_data_local, get_data_gcp
from BatteryProject.ModelOne.preprocessing import get_features_target
from BatteryProject.ModelOne.model_params import features




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
        self.pipeline = None
        self.raw_data = None
        self.target = None
        self.features = None


        # for MLFlow
        #self.experiment_name = EXPERIMENT_NAME

    def get_data(self, features_name):
        self.features_name = features_name

        df_dict = {}
        for name, path in self.features_name.items():
            df = get_data_local(path)
            df_dict[name] = df

        self.raw_data = df_dict
        self.features, self.target = get_features_target(self.raw_data, deep = self.deep, classes = self.classes)

        return self

    def set_pipeline(self):
        pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('model', LogisticRegression(max_iter = 1000))])

        self.pipeline = pipe

        return self

    def run(self, grid_params):
        """ Run a Grid Search on the grid search params """
        self.grid_params = grid_params
        gs_results = GridSearchCV(self.pipeline, self.grid_params, n_jobs = -1, cv = 5)
        gs_results.fit(self.features, self.target)

        self.grid_search = gs_results

        return self


    def save_model(reg):
        """ method that saves the model into a .joblib file and uploads it on Google Storage /models folder """
        pass

if __name__ == '__main__':
    trainer = Trainer()
