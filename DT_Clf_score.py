from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,auc

from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core import Run

import numpy as np
import pandas as pd

import json
import os
import logging
import joblib

def init():
    global model
    model_path = os.path.join( os.getenv("../AzureMLModel"), "./DTmodel.pkl")
    model = joblib.load('./DTmodel.pkl')
    logging.info("Init complete")

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    y_hat = model.predict(data)
    return y_hat.tolist()