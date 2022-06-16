from pyexpat import model
from google.cloud import storage
import numpy as np
import pandas as pd
from sklearn import metrics


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


    def save_model(reg):
        """ method that saves the model into a .joblib file and uploads it on Google Storage /models folder """
        pass

if __name__ == '__main__':
    trainer = Trainer()
