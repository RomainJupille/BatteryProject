
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from timeit import default_timer as timer

from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, optimizers, callbacks
from sklearn.model_selection import train_test_split, learning_curve

from memoized_property import memoized_property
from tensorflow.keras.metrics import RootMeanSquaredError
from BatteryProject.data import get_data_local
from BatteryProject.ModelTwo.get_features import get_features_target
from BatteryProject.ModelTwo.scaler import CustomStandardScaler
from BatteryProject.ModelTwo.loss import root_mean_squared_error



from BatteryProject.params import MLFLOW_URI
import mlflow


class Trainer():

    def __init__(self, features_name = None, deep = 20, offset = 15):
        self.features_name = features_name
        self.n_features = len(features_name)
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

        index_array = np.arange(df_dict['disc_capa'].shape[0]) # [0, ... 134]

        self.train_index, self.test_index = train_test_split(index_array , test_size = 0.2, random_state=0)
        self.train_index, self.val_index = train_test_split(self.train_index , test_size = 0.25, random_state=0)

        self.X_train, self.y_train = get_features_target(self.raw_data, self.deep, self.offset, self.train_index)
        self.X_val, self.y_val = get_features_target(self.raw_data, self.deep, self.offset, self.val_index)
        self.X_test, self.y_test = get_features_target(self.raw_data, self.deep, self.offset, self.test_index)
        #self.X, self.y = get_features_target(self.raw_data, offset = self.offset, deep = self.deep)
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.3)

        return self

    def get_baseline(self):
        df = self.raw_data['disc_capa'].iloc[self.train_index,:].copy()
        results = pd.DataFrame()
        for i in range(0,3000,50):
            val = (3000 - i) - df[df.iloc[:,i].isna() == False].isna().sum(axis=1).mean()
            results[i] = [val]

        results = results.T.reset_index()
        results.columns = ['range', 'mean']
        results.fillna(0, inplace= True)

        test = self.X_test[:,self.deep-1,self.n_features]
        prediction = []
        for i in range(test.shape[0]):
            index = test[i]
            pred = (results['range'] - index).abs().argsort()[0]
            pred = results.iloc[pred,1]
            prediction.append(pred)

        self.prediction = prediction

        baseline = root_mean_squared_error(self.y_test, prediction)
        self.baseline = baseline

        return self



    def initialize_model(self):
        """ CNN1D model """
        model = models.Sequential()

        # layer de convolution avec x filtres
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

        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu')) # intermediate layer
        model.add(layers.Dense(1, activation='linear'))

        model.compile(loss='mse',
                      optimizer=optimizers.Adam(learning_rate = 0.001),
                      metrics=['mse'], # 'mae', mse', 'rmse', 'rmsle'
                      )

        self.model = model
        return model


    def scaling(self, scaler = CustomStandardScaler()):
        """self.mean_scaler = self.X_train.mean(axis=0).reshape(1,self.deep,self.n_features +1)
        self.std_scaler = self.X_train.std(axis=0).reshape(1,self.deep,self.n_features  +1)
        self.X_train_scaled = (self.X_train - self.mean_scaler) / self.std_scaler
        self.X_val_scaled = (self.X_val - self.mean_scaler) / self.std_scaler
        self.X_test_scaled = (self.X_test - self.mean_scaler) / self.std_scaler"""

        self.scaler = scaler
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(self.X_train)
        print(self.X_train_scaled)

        return self


    def set_pipeline(self):
        model = models.Sequential()

        # layer de convolution avec x filtres
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

        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu')) # intermediate layer
        model.add(layers.Dense(1, activation='linear'))

        self.model = model
        return self


    def run(self,
            opt = 'rmsprop',
            loss = 'mse',
            metrics = [RootMeanSquaredError()],
            epochs = 500,
            batch_size = 32):

        starttime = timer()

        self.optimizer = opt
        self.loss = loss
        self.metrics = metrics
        self.optimizer = opt
        self.epochs = epochs
        self.bacth_size = batch_size

        es = callbacks.EarlyStopping(patience = 10, restore_best_weights= True)

        self.model.compile(
            self.optimizer,
            loss = self.loss,
            metrics = self.metrics)

        self.history = self.model.fit(
            self.X_train_scaled,
            self.y_train,
            epochs = self.epochs,
            batch_size=self.bacth_size,
            validation_data= (self.X_val_scaled,self.y_val),
            callbacks=[es])

        self.training_time = timer() - starttime

        return self


    """def fit(self):
        es = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        self.history = self.model.fit(self.X_train_scaled, self.y_train,
                                 batch_size=16,
                                 epochs=1000,
                                 validation_data = (self.X_val_scaled,self.y_val),
                                 callbacks=[es],
                                 verbose=1)"""

    def eval(self):
        #self.score = accuracy_score(self.grid_search.best_estimator_.predict(self.X_test), self.y_test)
        error = self.model.evaluate(self.X_test_scaled, self.y_test, verbose=0)
        return error

    def predict(self, features_to_predict):
        return self.model.predict(features_to_predict)


    def create_save_id_model(self):
        dir_path = os.path.join(os.path.dirname(__file__),'Models','IDs.csv')
        df_ids = pd.read_csv(dir_path)
        last_id = df_ids.values.max()
        new_id = last_id + 1

        with open(dir_path, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([new_id.tolist()])
        if new_id < 10:
            self.ID = f"000{new_id}"
        elif new_id < 100:
            self.ID = f"00{new_id}"
        elif new_id < 1000:
            self.ID = f"0{new_id}"
        else:
            self.ID = f"{new_id}"
        return self.ID


    def save_model_locally(self):
        """Save the model into a .joblib format"""

        joblib.dump(self.model, f'BatteryProject/ModelTwo/Models/model_{self.ID}.joblib')
        filename = f'model_{self.ID}.joblib'
        dir_path = os.path.dirname(__file__)
        joblib.dump(self.model, os.path.join(dir_path,'..','..', filename))
        return self


    def save_model(self):
        """Save the model into a .joblib format"""

        #get the ID
        self.create_save_id_model()

        #Save in MLFlow
        EXPERIMENT_NAME = f"[FR] [Marseille] [BatteryTeam] Battery_ModelTwo + 1"
        self.experiment_name = EXPERIMENT_NAME
        self.mlflow_log_param("ID", self.ID)
        self.mlflow_log_param("deep", self.deep)
        self.mlflow_log_param("offset", self.offset)
        self.mlflow_log_param("unit_type", self.unit_type)
        self.mlflow_log_param("n_units", self.n_units)
        self.mlflow_log_param("dropout_rate", self.dropout)
        self.mlflow_log_param("dropout_layer", self.dropout_layer)

        for feature in list(self.features_name.keys()):
            self.mlflow_log_param(feature, 1)

        self.mlflow_log_metric("baseline", self.baseline)
        self.mlflow_log_metric("train_eval", self.eval_results['eval_train'])
        self.mlflow_log_metric("validation_eval", self.eval_results['eval_val'])
        self.mlflow_log_metric("test_eval", self.eval_results['eval_test'])
        self.mlflow_log_metric("training_time", self.training_time)
        self.mlflow_log_metric("epochs", len(self.history.history['loss']))

        self.save_model_locally()
        return self


    def plot_mse(self, title=None):
        """ """
        if not self.history:
            print("model must be fitted before")
            return
        history = self.history
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,4))
        ax1.plot(history.history['loss'])
        ax1.plot(history.history['val_loss'])
        ax1.set_title('Model loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylim(ymin=0, ymax=500000)
        ax1.legend(['Train', 'Validation'], loc='best')
        ax1.grid(axis="x",linewidth=0.5)
        ax1.grid(axis="y",linewidth=0.5)
        ax2.plot(history.history['root_mean_squared_error'])
        ax2.plot(history.history['val_root_mean_squared_error'])
        ax2.set_title('RMSE')
        ax2.set_ylabel('RMSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylim(ymin=0, ymax=500)
        ax2.legend(['Train', 'Validation'], loc='best')
        ax2.grid(axis="x",linewidth=0.5)
        ax2.grid(axis="y",linewidth=0.5)
        plt.show()


    def gen_one_cell(self, type="medium", offset=0):
        """
            return one reference cell (for prediction)
            (1,deep,features)
        """
        bad_cell_range_max = 500
        good_cell_range_min = 1000

        if type == "bad":
            mask = self.y_test < bad_cell_range_max
        elif type == "medium":
            mask = (self.y_test >= bad_cell_range_max) & (self.y_test < good_cell_range_min)
        elif type == "good":
            mask = self.y_test >= good_cell_range_min

        X_test = self.X_test[mask]
        y_test = self.y_test[mask]
        id = np.random.randint(0,len(y_test))
        X_test_scaled = self.scaler.transform(np.array([X_test[id]]))
        return X_test_scaled, y_test[id]


# MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return mlflow.tracking.MlflowClient()

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
    trainer = Trainer()
