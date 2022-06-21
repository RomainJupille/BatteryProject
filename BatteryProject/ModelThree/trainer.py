from telnetlib import SE
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import RootMeanSquaredError

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split, learning_curve
from BatteryProject.data import get_data_local
from BatteryProject.ModelThree.get_features import get_features_target
from BatteryProject.ModelThree.loss import root_mean_squared_error

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

    def set_pipeline(self):
        model = Sequential()
        model.add(LSTM(units = 4, activation = 'tanh', return_sequences=True))
        model.add(LSTM(units = 4, activation = 'tanh'))
        model.add(Dense(20, activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation = 'linear' ))

        self.model = model
        return self

    def run(self,
            opt = 'rmsprop',
            loss = 'mse',
            metrics = [RootMeanSquaredError()],
            epochs = 100,
            batch_size = 32):

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
            'res_train' : res_train,
            'res_val' : res_val,
            'res_test' : res_test
        }

        return eval_dict


    def save_model(reg):
        """To be done"""
        pass

if __name__ == '__main__':
    features = {
        'disc_capa' : 'summary_discharge_capacity.csv',
        'dis_ener' : 'summary_discharge_energy.csv',
        'temp_avg' : 'summary_temperature_average.csv',
        'char_capa' : 'summary_charge_capacity.csv'}
    trainer = Trainer(features_name=features)
    trainer.get_data()
    trainer.set_pipeline()
    trainer.run()

    '''
    Params à faire varier
    -> deep
    -> offset
    -> nb de layer
    '''
