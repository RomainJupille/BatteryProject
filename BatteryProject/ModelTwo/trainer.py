
from google.cloud import storage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from BatteryProject.data import get_data_local, get_data_gcp
from BatteryProject.ModelTwo.get_features import get_features_target
from BatteryProject.ModelTwo.model_params import features

from tensorflow.keras import models, layers, optimizers, callbacks



class Trainer():

    def __init__(self, features_name = None, deep = 5, offset = 0, classes = [550], grid_params ={} ):
        """
            features : list of features (in the shape of a dictionnary)
        """
        self.features_name = features_name
        self.deep = deep
        self.offset = offset
        self.classes = classes
        self.target_name = 'disc_capa'
        self.grid_params = grid_params
        self.pipeline = None
        self.history = None
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
        self.X, self.y = get_features_target(self.raw_data, offset = self.offset, deep = self.deep, classes = self.classes)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.3)

        return self


    def initialize_model_CNN(self):
        """ CNN1D model """
        timesteps = self.deep
        n_features = 4

        model = models.Sequential()

        # layer de convolution avec 16 filtres
        model.add(layers.Conv1D(32,
                                kernel_size=4,
                                strides=1,
                                padding='same',
                                activation="relu",
                                input_shape=self.X_train.shape[1:]))
        model.add(layers.MaxPool1D(pool_size=2))

        model.add(layers.Conv1D(32,
                                kernel_size=3,
                                padding='same',
                                activation="relu"))
        model.add(layers.MaxPool1D(pool_size=2))

        model.add(layers.Conv1D(32,
                                kernel_size=2,
                                padding='same',
                                activation="relu"))
        model.add(layers.MaxPool1D(pool_size=2))

        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu')) # intermediate layer
        model.add(layers.Dense(1, activation='linear'))

        # optimizer (ATTENTION: fait planter si utilisé avec ces parametres)
        #adam_opt = optimizers.Adam({'learning_rate':0.01, 'beta_1':0.9, 'beta_2':0.999})

        # regression
        model.compile(loss='mse',
                      optimizer=optimizers.Adam(),
                      metrics=['mse'], # 'mae', mse', 'rmse', 'rmsle'
                      )
        return model


    # TODO: à terminer
    def initialize_model_RNN(self):
        pass


    def fit(self, model):
        es = callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        self.history = model.fit(self.X_train, self.y_train,
                                 batch_size=16,
                                 epochs=100,
                                 validation_split=0.3,
                                 callbacks=[es],
                                 verbose=1)


    def set_pipeline_CNN(self, scaler = RobustScaler()):
        model = self.initialize_model_CNN()
        pipe = Pipeline([
            ('scaler', scaler),
            ('model', model)])
        self.pipeline = pipe
        return self


    # TODO: à terminer
    def set_pipeline_RNN(self, scaler = RobustScaler()):
        model = self.initialize_model_RNN()
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


    def eval(self, model):
        #self.score = accuracy_score(self.grid_search.best_estimator_.predict(self.X_test), self.y_test)
        error = model.evaluate(self.X_test, self.y_test, verbose=0)
        return error

    def predict(self, model, row_to_predict):
        return model.predict(row_to_predict)

    def save_model(self, reg):
        """ method that saves the model into a .joblib file and uploads it on Google Storage /models folder """
        pass

    def plot_history(self, title=None):
        """ """
        if not self.history:
            print("model must be fitted before")
            return
        history = self.history

        fig, ax = plt.subplots(1,2, figsize=(20,7))

        # --- LOSS ---
        ax[0].plot(history.history['loss'])
        ax[0].plot(history.history['val_loss'])
        ax[0].set_title('Model loss')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylim((0,3))
        ax[0].legend(['Train', 'Test'], loc='best')
        ax[0].grid(axis="x",linewidth=0.5)
        ax[0].grid(axis="y",linewidth=0.5)

        if title:
            fig.suptitle(title)

if __name__ == '__main__':
    trainer = Trainer()
