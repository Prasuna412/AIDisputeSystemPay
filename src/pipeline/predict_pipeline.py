import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            pred_prob = model.predict_proba(data_scaled)
            return pred, pred_prob
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, text: str):
        self.text = text

    def get_data(self):
        try:
            text = [self.text]
            return text

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    text = ["I lost my card"]
    model = PredictPipeline()
    print(model.predict(text))
