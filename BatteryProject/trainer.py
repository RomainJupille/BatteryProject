from google.cloud import storage
import numpy as np
import pandas as pd


class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        # for MLFlow
        #self.experiment_name = EXPERIMENT_NAME

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
    """ runs a training """
