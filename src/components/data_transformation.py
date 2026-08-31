import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from .preprocessor import TextCleaner, TextVectorizer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    '''
    To store the preprocessor at defined location
    '''
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    '''
    Responsible to convert text into vectors
    '''

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This Function is responsible for data transformation.
        Input should be the detailed dispute discussion textual column
        '''
        try:
            preprocessor = Pipeline([
                ('cleaner', TextCleaner()),
                ('vectorizer', TextVectorizer()),
                ('scaler', MinMaxScaler())
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_pickle(train_path)
            test_df = pd.read_pickle(test_path)

            logging.info("Reading Train and Test Data Completed")

            logging.info("Getting Preprocessing Object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "classes"
            input_column = "text"
            input_feature_train_df = train_df[input_column]
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df[input_column]
            target_feature_test_df = test_df[target_column]

            logging.info(
                "Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)
            logging.info(
                "Text is processed and converted into sentence embeddings which can be used to train/test the ML model")
            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
