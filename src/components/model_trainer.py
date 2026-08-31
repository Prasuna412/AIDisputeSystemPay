import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path=None):
        try:
            logging.info("Splitting Input as Traning and Testing Data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest Classifier": RandomForestClassifier(),
                "Support Vector Machine": SVC(),
                "XG Boost": XGBClassifier(),
                "Naive Bayes": GaussianNB(),
                "K Nearest Neigbors": KNeighborsClassifier()
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            logging.info("Model Found")
            best_model = models['Random Forest Classifier']

            # Can load preprocessor if we are getting new data
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model

            )

            predicted = best_model.predict(X_test)

            return (
                accuracy_score(y_test, predicted),
                precision_score(y_test, predicted, average='weighted'),
                recall_score(y_test, predicted, average='weighted'),
                f1_score(y_test, predicted, average='weighted')
            )

        except Exception as e:
            raise CustomException(e, sys)
