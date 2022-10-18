# Introduction
This work is a data-science study project achieved in 2022 by a 4 students team.

The dataset used during the project comes from a Nature Energy article: https://energy.stanford.edu/sites/g/files/sbiybj9971/f/346501527888d799d94279cfe74049fde53ca2d5a1275d083d28f925253cf657.pdf.

The dataset can be downloaded there: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204

The dataset is composed of measurements from a set of 140 batteries that have been fully charged and discharged until they fail.
During the process several physical and electrical measures have been saved (such as temperature, charge and discharge capacity or internal resistance)
To get a sens of the data:
- The average number of life-cycles is 852
- The maximum is 2239
- The minimum is 1 (the problem came from the data acquisition during experiment)

The orignal dataset is composed of:
- 'Summary features': One measurement per charge/discharge cycle (aggregated information per cycle)
- 'Interpolated features: '1000 measurements per cycle', with continuous measure through cycles.
This project only deals with 'summary features'.

# Presentation of project
The goal of the project is to:
 1/ Create a binary classification model that predicts if a battery will reach a given treshold of life-cycles based on the observation of some first cycles.
 2/ Create a regression model predicting the number of life-cycle left to the battery before it fails based on the observation of its previous cycles.

The project is composed of four steps
- Extraction, transformation and cleaning of the orignal dataset in a proper format for next steps
- Exploration of the data and of the features
- Construction and testing of the model one (binary classification)
- Construction and testing of the model two (prediction of life-cycles left to the battery)

# Description of the pyhton project

## Overview
- The main directory is organized as such:
.
- BatteryProject
    - BatteryProject
    - BatteryProject.egg-info
    - build
    - notebooks
    - scripts
    - tests
- InitialData
- TransformedData

The BatteryProject directory contains the python package (given in this github repo)
The two other directories should be added :
- The InitialData directory contains the raw data (json files) from the article (and downloadable in the article)
- The TransformedData dir contrains the clean and transformed data that will be used in the models

## Data preparation
The data preparation is achieved by the 3 python files in the BatteryProject directory
- BatteryProject
    - BatteryProject
      - data_transforming.py
      - get_feature.py
      - transform_params.py

The data can be extracted by running the data_transforming.py file.

Summary of the process achieved by those files:
- In the raw data, each JSON file contains all the measurement of one battery.
  After data transformation, each file correponds to a feature, aggregating data from all batteries.
  In the transformedn files, 1 file = 1 feature (with the barcode to keep the battery id)
- Some measurement have failed during the acquisition process (the information is given in the article).
  The method ignores those batteries
- Some batteries measurement have been split onto 2 differents files.
  The method aggregates the measurement coming from a similar battery
- The method has been designed to handle 'Interpolated files'
  However be aware that those files are heavy and long to procces (for a personal desk computer at least)
- Additional file containing all the parameters fixed of the batteries has been added too (including methods of charge/discharge)

The notebook '01_data_extraction_step_by_step' shows what the method is doing step by step

3/ Data exploration
- The data exploration is achieved in the notebook '02_data_exploration'
- The notebook also gives a visual sens of the two models that will be build in the project



## Model One: Binnary classification

### Basics of the model
- Batteries are split into 2 categories: the ones that reached 550 life-cycles and the ones that didn't reach it. This classification is the
  target of the model
- Features used for the training are the measurements of the 5 first life-cycles (which is called 'deep' in the code)
- The deep is set to 5 to correspond to the Nature paper. However this can be changed to anything else.

### Models tested:
- Several models have been tested (Logistic and RandomForest, with different hyperparameters, differents scalers and different input features)
- Both the 550 treshold and the 5 cycles deep can be changed

### Overview of the code
The code for this model is contained in the ModelOne directory:
.
- data
- models
- best_model
- trainer.py
- get_features.py
- __init__.py
- model_params.py

Description of the python files:
- model_params.py contains the list of hyper-parameters of the model to be tested when running the model
- get_feature.py contrains the method to get the data from the TransformedData directory
- trainer.py contains
  -> the class definition and methods of the trainer class used to train, test and saved different models
  -> the __main__ code that should be run to launch the model training

When run, the trainer.py __main__ code achieve the following steps:
- get all the combination of model arcgitecture and hyper-parameters from the 'model_params.py' file
- test all the [features, model, scaler] combinations in a loop using the trainer class. For each trio:
  -> create a train class
  -> get the features and create a train/test split
  -> set the pipeline
  -> train the model with a grid_search (testing all hyper-parameters)
  -> evaluate the model (accuracy, precision, roc_aux) with the grid-search
  -> save the model results and performance (feature used, model, scaler, hyper-parameters, performance metrics ) in models_record.csv file in the best_model directory
  -> save the data to run the final testing (X_test, y_test, model.joblib) in the repository data
  -> save the model.joblib (in the models directory)


To explore the trainer class, refer to the notebook '03_test_model_one_trainer_class' directory

### Results
(The data of the best model are stored along with the result records are available in the 'best_model' directory)

The model has been tested on the test set. results are as follow:
- **Accuracy of the model: 0.9722222222222222**
- **Precision of the model: 0.9565217391304348**
- **Roc auc of the model: 0.9642857142857143**
- The best model is a logisticregression model that uses only 4 features
    Pipeline(steps=[('scaler', StandardScaler()),
      ('model',
      LogisticRegression(C=5.0, max_iter=500, penalty='l1',
      solver='liblinear'))])


## Model two: regression model

### Basics of the model
The targer of this model is to predict how many life-cycle remains to the battery betfore it fails.
For this purpose we construct a new sample set from the initial data
- Features are composed of the measurement certain amount of life-cycle, which we call the 'deep' in the code
  The deep is between 20 to 40.
- The target of the model is the remaining number of life-cycles, after the last cycle used in the feature
- For a given battery, there are several samples (at different stages of the life of the battery). Different samples are separated of a
  certain amount of cycles (which we call the 'offset' in the code)-

### Model description
- The model is a RNN network with several layers and units
- LSTM only have been tested during the project however it can be changed to GRU in the paremeters)
- The overfitting is manged with early stopping and dropout
- We tested a RNN model with different configurations
  -> different number of layers from 3 and 4
  -> different number of units in each cell from 4 and 5
  -> different dropout rate

### Results
- The best model performance
  -> rmse of 211 (which means that the battery make an everage error or 221 life-cycles)
  -> mean percentage error of 21.8%
- The model performs better for battery with lower number of life-cycles left
  -> The mean percentage error for samples with less than 600 cycles left is 19.7%
  -> The error is 27% for batteries with more than 600 cyles

Parameters of the best model
- Unit type              LSTM
- Number of units        5
- Number of layers       2
- Dropout rate           0.2
