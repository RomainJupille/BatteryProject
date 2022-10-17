import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from timeit import default_timer as timer

from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from BatteryProject.get_feature import get_data_local
from BatteryProject.ModelTwo.get_features import get_features_target
from BatteryProject.ModelTwo.loss import root_mean_squared_error
from BatteryProject.ModelTwo.model_params import *


class Trainer():

    def __init__(self, features_name = None, deep = 30, offset = 20):
        #list of the features used in the network
        self.features_name = features_name
        self.n_features = len(features_name)

        #lenght of the data used to train the model
        self.deep = deep

        #distance between 2 set of data
        self.offset = offset
        self.target_name = 'disc_capa'

    def get_data(self):
        '''
        Class method that splits the dataset into training, validation and test sets
        The method then extract X's, y's and barcode's sets using the get_features_target method
        '''
        df_dict = {}
        #get all the raw data corresponding to feature names
        for name, path in self.features_name.items():
            df = get_data_local(path)
            df_dict[name] = df

        self.raw_data = df_dict

        #spliting the data indexes into 3 sub-sets
        self.train_index, self.test_index = train_test_split(np.arange(df_dict['disc_capa'].shape[0]) , test_size = 0.2, random_state=1)
        self.train_index, self.val_index = train_test_split(self.train_index , test_size = 0.25, random_state=1)

        self.split_indexes = {}
        self.split_indexes['train'] = self.train_index
        self.split_indexes['val'] = self.val_index
        self.split_indexes['test'] = self.test_index




        #get the data and store them into class variables
        self.X_train, self.y_train, self.bc_train = get_features_target(self.raw_data, self.deep, self.offset, self.train_index)
        self.X_val, self.y_val, self.bc_val = get_features_target(self.raw_data, self.deep, self.offset, self.val_index)
        self.X_test, self.y_test, self.bc_test = get_features_target(self.raw_data, self.deep, self.offset, self.test_index)

        return self

    def get_baseline(self):
        '''
        Definition of a score baseline.
        For a given dataset, the baseline model predicts the remaining number of life-cycles as the mean of remaining cycles
           for all batteries that reach the current state of the dataset.
        The score used for the baseline is the root_mean_squared error.
        '''
        #get the target data (only using train set)
        df = self.raw_data['disc_capa'].iloc[self.train_index,:].copy()
        results = pd.DataFrame()

        #every 50 cycles, measure the mean of remaining cycles for batteries that reach this number of cycles
        for i in range(0,3000,50):
            val = (3000 - i) - df[df.iloc[:,i].isna() == False].isna().sum(axis=1).mean()
            results[i] = [val]
        results = results.T.reset_index()
        results.columns = ['range', 'mean']
        results.fillna(0, inplace= True)

        #get the last 'disc capa' measure from all X_test samples
        test = self.X_test[:,self.deep-1,self.n_features]
        prediction = []
        for i in range(test.shape[0]):
            #get the index of the measurement
            index = test[i]

            #get the closest index
            pred_index = (results['range'] - index).abs().argsort()[0]
            #get the prediction from the baseline model based on the index
            pred = results.iloc[pred_index,1]
            prediction.append(pred)

        self.prediction = prediction

        #measure baseline
        baseline = root_mean_squared_error(self.y_test, self.prediction)
        self.baseline = baseline

        return self

    def scaling(self):
        '''
        Scaling of X's sets
        The scaler is fit with X_train only
        '''
        self.mean_scaler = self.X_train.mean(axis=0).reshape(1,self.deep,self.n_features +1)
        self.std_scaler = self.X_train.std(axis=0).reshape(1,self.deep,self.n_features  +1)

        self.X_train_scaled = (self.X_train - self.mean_scaler) / self.std_scaler
        self.X_val_scaled = (self.X_val - self.mean_scaler) / self.std_scaler
        self.X_test_scaled = (self.X_test - self.mean_scaler) / self.std_scaler

        return self

    def set_pipeline(self, unit_type = 'LSTM', n_layer = 1, n_unit = 1, dropout = 0.0, dropout_layer = True):
        '''
        Definition of the RNN model
        '''
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
        '''
        Compile and fit the RNN model
        '''
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
        '''
        Evaluate the model
        '''
        res_train = self.model.evaluate(self.X_train_scaled, self.y_train, batch_size=None)[1]
        res_val = self.model.evaluate(self.X_val_scaled, self.y_val, batch_size=None)[1]

        eval_dict = {
            'eval_train' : res_train,
            'eval_val' : res_val
        }

        self.eval_results = eval_dict

        return eval_dict

    def save_model(self):
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'best_model','models_record.csv')
        df_records = pd.read_csv(dir_path)
        try:
            df_records = df_records.drop(columns='Unnamed: 0')
        except:
            pass

        if df_records.shape[0] == 0:
            print("1st row added to the model")
            print('\n')
            new_id=1
        else:
            last_id = df_records['Try_ID'].values.max()
            new_id = last_id + 1
            print("adding a row in the model")
            print('\n')

        data = pd.DataFrame()
        data['Try_ID'] = [new_id]
        data["deep"] = [self.deep]
        data["offset"] = [self.offset]

        data["HyperParams_unit_type"] = [self.unit_type]
        data["HyperParams_n_units"] = [self.n_units]
        data["HyperParams_n_layer"] = [self.n_layer]
        data["HyperParams_dropout_rate"] = [self.dropout]
        data["HyperParams_dropout_layer"] = [self.dropout_layer]

        for key in self.features_name.keys():
            data[f"Features_{key}"] = ['X']

        data["Metrics_baseline"] = [self.baseline]
        data["Metrics_train_eval"] = [self.eval_results['eval_train']]
        data["Metrics_validation_eval"] = [self.eval_results['eval_val']]
        data["Metrics_training_time"] = [self.training_time]
        data["Metrics_epochs"] = [len(self.history.history['loss'])]


        print("==== data added to the df======")
        print(data)
        print('\n')
        df_records = df_records.append(data, ignore_index=True)

        col_list = df_records.columns
        col_l1 = ['Try_ID',  'deep', 'offset']
        col_l2 = [a for a in col_list if a[0:8] == 'Features']
        col_l3 = [a for a in col_list if a[0:7] == 'Metrics']
        col_l4 = [a for a in col_list if a[0:11] == 'HyperParams']

        new_col_list = col_l1 + col_l2 + col_l3 + col_l4
        df_records = df_records[new_col_list]

        print("saving the record")
        print(df_records)
        print("\n")
        df_records.to_csv(dir_path)

        if new_id <= 9:
            self.ID = f"000{new_id}"
        elif new_id < 99:
            self.ID = f"00{new_id}"
        elif new_id < 999:
            self.ID = f"0{new_id}"
        else:
            self.ID = f"{new_id}"

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models', f"model_{self.ID}.joblib")
        joblib.dump(self.model, model_path)

        return self

    def save_data(self):
        #raw_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"raw_data_{self.ID}.csv")
        #train_split_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"train_split_index_{self.ID}.csv")
        #val_split_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"val_split_index_{self.ID}.csv")
        #test_split_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"test_split_index_{self.ID}.csv")
        X_test_scaled_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"X_test_scaled{self.ID}.csv")
        y_test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"y_test_{self.ID}.csv")
        bc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"bc_{self.ID}.csv")

        #self.split_indexes.to_csv(split_index_path)
        #np.savetxt(train_split_index_path , self.train_index, delimiter=",")
        #np.savetxt(val_split_index_path , self.val_index, delimiter=",")
        #np.savetxt(test_split_index_path , self.test_index, delimiter=",")

        np.savetxt(X_test_scaled_path , self.X_test_scaled.reshape(self.X_test_scaled.shape[0], -1), delimiter=",")
        np.savetxt(y_test_path, self.y_test, delimiter=",")
        np.savetxt(bc_path , self.bc_test, delimiter=",", fmt="%s")

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
                                t.split_indexes = trainer_data.split_indexes
                                t.train_index = trainer_data.train_index
                                t.val_index = trainer_data.val_index
                                t.test_index = trainer_data.test_index

                                t.X_train = trainer_data.X_train
                                t.bc_test = trainer_data.bc_test
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
                                t.save_data()
