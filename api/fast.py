
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import unquote

import joblib
import os

import numpy as np
import pandas as pd
from datetime import datetime
import pytz


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def serialize_features(X):
    serial = X.flatten()
    return ",".join([str(i) for i in serial])

def deserialize_features(X_serialized, n_features=5, deep=20, delimiter=','):
    res = np.array([float(idx) for idx in X_serialized.split(delimiter)])
    return res.reshape(1,deep,n_features)



@app.get("/")
def index():
    return {"greeting": "Hello world"}



# prediction model1
@app.get("/predict1")
def predict(n_features, deep, X_val_serialized):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model = joblib.load(os.path.join(dir_path, "..", "model1.joblib"))

    response = {
        "predict": 0,
    }
    return response


# prediction model2
@app.get("/predict2")
def predict(n_features, deep, X_val_serialized):
    X_val = deserialize_features(unquote(X_val_serialized),
                                 n_features=int(n_features),
                                 deep=int(deep))

    # récupère le pipeline entrainé (en local)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model = joblib.load(os.path.join(dir_path, "..", "model2.joblib"))
    y_pred = model.predict(X_val)[0]

    print("prediction model2:", int(y_pred[0]))

    response = {
        "predict": int(y_pred[0]),
    }
    return response


# prediction model3
@app.get("/predict3")
def predict(n_features, deep, X_val_serialized):
    X_val = deserialize_features(unquote(X_val_serialized),
                                 n_features=int(n_features),
                                 deep=int(deep))

    # récupère le pipeline entrainé (en local)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model = joblib.load(os.path.join(dir_path, "..", "model3.joblib"))
    y_pred = model.predict(X_val)[0]

    print("prediction model3:", int(y_pred[0]))

    response = {
        "predict": int(y_pred[0]),
    }
    return response
