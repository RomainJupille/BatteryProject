import numpy as np


from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt

from BatteryProject.data import get_data_local
from BatteryProject.ModelThree.get_features import get_features_target

class Trainer():

    def __init__(self, features_name = None, deep = 5, offset = 15):
        """
        features : list of features (in the shape of a dictionnary)
        """
        self.features_name = features_name
        self.deep = deep
        self.offset = offset
        self.target_name = 'disc_capa'

    def get_data(self):
        '''To be done'''
        df_dict = {}
        for name, path in self.features_name.items():
            df = get_data_local(path)
            df_dict[name] = df

        self.raw_data = df_dict
        self.train_index, self.test_index = train_test_split(np.array(df_dict['disc_capa'].shape[0]) , test_size = 0.3)

        self.X_train, self.y_train = get_features_target(self.raw_data, self.deep, self.offset, self.train_index)

        self.X_test, self.y_test = get_features_target(self.raw_data, self.deep, self.offset, self.train_index)

        return self

    def set_pipeline(self):
        '''to be done'''
        return self

    def run(self, grid_params):
        """To be done"""
        return self

    def print_learning_curve(self):
        '''To be done'''
        # self.best_model = self.grid_search.best_estimator_["model"]
        # train_sizes = list(range(20,108,10))
        # train_sizes, train_scores, test_scores = learning_curve(
        # self.best_model, X=self.X, y=self.y, train_sizes=train_sizes, cv=5, n_jobs=-1)
        # train_scores_mean = np.mean(train_scores, axis=1)
        # test_scores_mean = np.mean(test_scores, axis=1)
        # plt.figure(figsize=(15,7))
        # plt.plot(train_sizes, train_scores_mean, label = 'Training score')
        # plt.plot(train_sizes, test_scores_mean, label = 'Test score')
        # plt.ylabel('accuracy score', fontsize = 14)
        # plt.xlabel('Training set size', fontsize = 14)
        # plt.title('Learning curves', fontsize = 18, y = 1.03)

        # plt.legend()
        pass



    def eval(self):
        '''To be done'''
        pass


    def save_model(reg):
        """To be done"""
        pass

if __name__ == '__main__':
    trainer = Trainer()
