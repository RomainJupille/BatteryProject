# Introduction
This work is a data-science study project achieved in 2022 by a 4 students team.
As the project was not c

The dataset used during the project comes from a Nature Energy article:me https://energy.stanford.edu/sites/g/files/sbiybj9971/f/346501527888d799d94279cfe74049fde53ca2d5a1275d083d28f925253cf657.pdf.

The data are measurements from a set of 140 batteries that have been fully charged and discharged until they fail.
During the process several physical and electrical measures have been saved (such as temperature, charge and discharge capacity or internal resistance)
Batteries generally failled between 300 to 2000 cycles. In average they failled around 600 cycles.

The data set is composed of:
- 'Summary features': One measurement per charge/discharge cycle
- 'Interpolated features: '1000 measurements per cycle'
This project only deals with 'summary features'.

# Presentation of project
The goal of the project was to produce two results:
 1/ Create a binary classification model allowing to predict if a battery will reach a given treshold of life-cycles based on the observation of some first cycles.
 2/ Create a linear model predicting the number of life-cycle left to the battery before it fails based on the observation of its previous cycles.

The project is composed of four steps
- Extraction, transformation and cleaning of the data in a proper shape usable in a data-science project
- Exploration of the data and of the features
- Construction and testing of the model one (binary classification with a treshold)
- Construction and testing of the model two (linear prediction of cycles left)

# Description of the pyhton project
1/ Overview
- The main directory should be organized in such way:
.
├── BatteryProject
│   ├── BatteryProject
│   ├── BatteryProject.egg-info
│   ├── build
│   ├── notebooks
│   ├── scripts
│   └── tests
├── InitialData
└── TransformedData

The BatteryProject dir contains the python package (given in this github repo)
The trwo other directory should be added :
- The InitialData dir contains the raw data (json files) downloadable in the article
- The TransformedData dir contrains the clean and transformed data that will be used in the models

2/ Data preparation
The data preparation is achieved by the 3 python files in the BatteryProject directory
├── BatteryProject
│   ├── BatteryProject
│   │   ├── data_transforming.py
│   │   ├── get_feature.py
│   │   ├── transform_params.py

The data can be extracted by running the data_transforming.py file.

Summary of the process achieved by those file:
- In the raw data, each file contains all the measurement of one battery.
  The method aggregates each feature into a file
- Some measurement have failed during the acquisition process (the information is given in the article).
  The method ignire those batteries
- Some batteries measurement have been split onto 2 differents files.
  The method aggregates the measurement coming from a similar battery
- The method has been designed to handle 'Interpolated files'n
  However be aware that those files are heavy and long to precces (for a personal desk computer)
- One file = One feature (wioth the barcode to keep the battery id)
- Additional file containing all the parameters fixed of the battery (including methods of charge/discharge)

The notebook '01_data_extraction_step_by_step' shows what the method is doing step by step

3/ Data exploration
- The data exploration is achieved in the notebook '02_data_exploration'
- The notebook also gives a visual sens of the two models that will be build in the project
