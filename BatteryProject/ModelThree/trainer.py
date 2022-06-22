import csv
from aioitertools import dropwhile
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import joblib
from timeit import default_timer as timer

from memoized_property import memoized_property
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from xgboost import train

from BatteryProject.data import get_data_local
from BatteryProject.params import MLFLOW_URI
from BatteryProject.ModelThree.get_features import get_features_target
from BatteryProject.ModelThree.loss import root_mean_squared_error
from BatteryProject.ModelThree.model_params import *



class Trainer():

    def __init__(self, features_name = None, deep = 30, offset = 20):
        """
        features : list of features (in the shape of a dictionnary)
        """
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
        self.train_index, self.test_index = train_test_split(np.arange(df_dict['disc_capa'].shape[0]) , test_size = 0.2, random_state=1)
        self.train_index, self.val_index = train_test_split(self.train_index , test_size = 0.25, random_state=1)
        self.X_train, self.y_train = get_features_target(self.raw_data, self.deep, self.offset, self.train_index)
        self.X_val, self.y_val = get_features_target(self.raw_data, self.deep, self.offset, self.val_index)
        self.X_test, self.y_test = get_features_target(self.raw_data, self.deep, self.offset, self.test_index)

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

    def scaling(self):
        self.mean_scaler = self.X_train.mean(axis=0).reshape(1,self.deep,self.n_features +1)
        self.std_scaler = self.X_train.std(axis=0).reshape(1,self.deep,self.n_features  +1)

        self.X_train_scaled = (self.X_train - self.mean_scaler) / self.std_scaler
        self.X_val_scaled = (self.X_val - self.mean_scaler) / self.std_scaler
        self.X_test_scaled = (self.X_test - self.mean_scaler) / self.std_scaler

        return self

    def set_pipeline(self, unit_type = 'LSTM', n_layer = 1, n_unit = 1, dropout = 0.0, dropout_layer = True):

        self.unit_type = unit_type
        self.n_layer = n_layer
        self.n_units = n_unit
        self.dropout = dropout
        self.dropout_layer = dropout_layer

        params ={
            'units' : n_unit,
            'activation' : 'tanh',
            'dropout' : dropout
            }

        model = Sequential()

        if unit_type == 'LSTM':
            if n_layer == 1:
                model.add(LSTM(**params))
            if n_layer > 1:
                for i in range(n_layer - 1):
                    model.add(LSTM(**params, return_sequences = True))
                model.add(LSTM(**params))
        elif unit_type == 'GRU':
            if n_layer == 1:
                model.add(GRU(**params))
            if n_layer > 1:
                for i in range(n_layer - 1):
                    model.add(GRU(**params, return_sequences = True))
                model.add(GRU(**params))


        model.add(Dense(20, activation = 'relu'))
        if dropout_layer == True:
            model.add(Dropout(0.2))
        model.add(Dense(1, activation = 'linear' ))

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

        es = EarlyStopping(patience = 10, restore_best_weights= True)

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
        ax1.set_ylim(ymin=0, ymax=1000000)
        ax1.legend(['Train', 'Validation'], loc='best')
        ax1.grid(axis="x",linewidth=0.5)
        ax1.grid(axis="y",linewidth=0.5)
        ax2.plot(history.history['root_mean_squared_error'])
        ax2.plot(history.history['val_root_mean_squared_error'])
        ax2.set_title('MSE')
        ax2.set_ylabel('MSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylim(ymin=0, ymax=1000)
        ax2.legend(['Train', 'Validation'], loc='best')
        ax2.grid(axis="x",linewidth=0.5)
        ax2.grid(axis="y",linewidth=0.5)
        plt.show()

    def eval(self):
        #res = root_mean_squared_error(np.array([1,2,3]), np.array([10,4,5]))

        res_train = self.model.evaluate(self.X_train_scaled, self.y_train, batch_size=None)[1]
        res_val = self.model.evaluate(self.X_val_scaled, self.y_val, batch_size=None)[1]
        res_test = self.model.evaluate(self.X_test_scaled, self.y_test,batch_size=None)[1]

        eval_dict = {
            'eval_train' : res_train,
            'eval_val' : res_val,
            'eval_test' : res_test
        }

        self.eval_results = eval_dict

        return eval_dict

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

    def save_model(self):
        """Save the model into a .joblib format"""

        #get the ID
        self.create_save_id_model()

        #Save in MLFlow
        EXPERIMENT_NAME = f"[FR] [Marseille] [BatteryTeam] Battery_ModelThree + 1"
        self.experiment_name = EXPERIMENT_NAME
        self.mlflow_log_param("ID", self.ID)
        self.mlflow_log_param("deep", self.deep)
        self.mlflow_log_param("offset", self.offset)
        self.mlflow_log_param("unit_type", self.unit_type)
        self.mlflow_log_param("n_units", self.n_units)
        self.mlflow_log_param("n_layer", self.n_layer)
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

        joblib.dump(self.model, f'BatteryProject/ModelThree/Models/model_{self.ID}.joblib')

        return self

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
    for feat in features:
        for val in deeps_offset:
            deep = val['deep']
            offset = val['offset']
            trainer_data = Trainer(features_name = feat, deep = deep, offset = offset)
            trainer_data.get_data()
            trainer_data.scaling()
            trainer_data.get_baseline()

            for unit_type in unit_types:
                for n_unit in n_units:
                    for n_layer in n_layers:
                        for drop in dropout:
                            for drop_layer in dropout_layer:
                                t = Trainer(features_name = feat, deep = deep, offset = offset)
                                t.raw_data = trainer_data.raw_data
                                t.X_train = trainer_data.X_train
                                t.X_test = trainer_data.X_test
                                t.X_val = trainer_data.X_val
                                t.X_train_scaled = trainer_data.X_train_scaled
                                t.X_test_scaled = trainer_data.X_test_scaled
                                t.X_val_scaled = trainer_data.X_val_scaled
                                t.y_train = trainer_data.y_train
                                t.y_test = trainer_data.y_test
                                t.y_val = trainer_data.y_val
                                t.baseline = trainer_data.baseline
                                t.set_pipeline(unit_type = unit_type, n_layer=n_layer, n_unit = n_unit, dropout = drop, dropout_layer= drop_layer)
                                t.run(epochs = 500)
                                t.eval()
                                t.save_model()
