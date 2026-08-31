import os
import sys

import numpy as np
import pandas as pd
from joblib import load

from .exception import CustomException
from .logger import logging

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)  # Training Model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = {"Accuracy": accuracy_score(y_train, y_train_pred), "Precision": precision_score(y_train, y_train_pred, average='weighted'),
                                 "Recall": recall_score(y_train, y_train_pred, average='weighted'), "f1 score": f1_score(y_train, y_train_pred, average='weighted')}
            test_model_score = {"Accuracy": accuracy_score(y_test, y_test_pred), "Precision": precision_score(y_test, y_test_pred, average='weighted'),
                                "Recall": recall_score(y_test, y_test_pred, average='weighted'), "f1 score": f1_score(y_test, y_test_pred, average='weighted')}

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        return load(file_path)
    except Exception as e:
        raise CustomException(e, sys)
