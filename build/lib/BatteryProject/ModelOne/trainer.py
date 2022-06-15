from google.cloud import storage
import numpy as np
import pandas as pd

from BatteryProject.data import get_data_local, get_data_gcp
from BatteryProject.ModelOne.preprocessing import get_features, get_target
from BatteryProject.ModelOne.model_params import features




class Trainer():

    def __init__(self, features_name = None, deep = 5, classes = [550]):
        """
            features : list of features (in the shape of a dictionnary)
        """
        self.pipeline = None
        self.features = features
        self.deep = deep
        self.classes = classes
        self.target_name = 'disc_capa'

        # for MLFlow
        #self.experiment_name = EXPERIMENT_NAME

    def get_data(self, features_name):
        self.features_name = features_name

        df_dict = {}
        for name, path in self.features_name.items():
            df = get_data_local(path)
            df_dict[name] = df

        self.raw_data = df_dict
        self.features = get_features(self.raw_data, self.deep)
        self.target = get_target(self.raw_data[self.target_name], self.classes)



        return self


    def preprocess(self, df):
        """ function that pre-processes the data """
        pass

    def set_pipeline(self):
        pass

    def train_model(X_train, y_train):
        """ function that trains the model """
        pass


    def save_model(reg):
        """ method that saves the model into a .joblib file and uploads it on Google Storage /models folder """
        pass


if __name__ == '__main__':
    trainer = Trainer()
