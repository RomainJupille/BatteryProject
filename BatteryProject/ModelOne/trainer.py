from google.cloud import storage
import numpy as np
import pandas as pd


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from BatteryProject.data import get_data_local, get_data_gcp
from BatteryProject.ModelOne.get_features import get_features_target
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
        self.X = None


        # for MLFlow
        #self.experiment_name = EXPERIMENT_NAME

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

    def set_pipeline(self, scaler = RobustScaler(), model = LogisticRegression(max_iter = 1000)):
        pipe = Pipeline([
            ('scaler', scaler),
            ('model', model)])

        self.pipeline = pipe

        return self

    def run(self, grid_params):
        """ Run a Grid Search on the grid search params """
        self.grid_params = grid_params
        gs_results = GridSearchCV(self.pipeline, self.grid_params, n_jobs = -1, cv = 5)
        gs_results.fit(self.X_train, self.y_train)

        self.grid_search = gs_results

        return self

    def eval(self):
        self.score = accuracy_score(self.grid_search.best_estimator_.predict(self.X_test), self.y_test)
        return self.score


    def save_model(reg):
        """ method that saves the model into a .joblib file and uploads it on Google Storage /models folder """
        pass

if __name__ == '__main__':
    trainer = Trainer()
