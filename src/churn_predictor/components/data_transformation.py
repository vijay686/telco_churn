import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn_predictor.exception import CustomException
from src.churn_predictor.logger import logging
from src.churn_predictor.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    train_arr_path = os.path.join('artifacts', "train_arr.npy")
    test_arr_path = os.path.join('artifacts', "test_arr.npy")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function creates the Scikit-Learn pipelines for numerical and categorical data.
        '''
        try:
            # TotalCharges is treated as numerical because we clean it before this step
            numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
            categorical_columns = [
                "gender", "SeniorCitizen", "Partner", "Dependents", 
                "PhoneService", "MultipleLines", "InternetService", 
                "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                "TechSupport", "StreamingTV", "StreamingMovies", 
                "Contract", "PaperlessBilling", "PaymentMethod"
            ]

            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(drop='first', handle_unknown='ignore'))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Data Cleaning: Handle the silent empty spaces in TotalCharges
            logging.info("Cleaning TotalCharges column")
            train_df['TotalCharges'] = train_df['TotalCharges'].replace(" ", np.nan).astype(float)
            test_df['TotalCharges'] = test_df['TotalCharges'].replace(" ", np.nan).astype(float)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Churn"
            # We drop customerID as it has no predictive power
            columns_to_drop = [target_column_name, "customerID"]

            input_feature_train_df = train_df.drop(columns=columns_to_drop)
            target_feature_train_df = train_df[target_column_name].map({'Yes': 1, 'No': 0})

            input_feature_test_df = test_df.drop(columns=columns_to_drop)
            target_feature_test_df = test_df[target_column_name].map({'Yes': 1, 'No': 0})

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # fit_transform on train, transform on test
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate features and targets into single numpy arrays for the model trainer
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the arrays to artifacts
            np.save(self.data_transformation_config.train_arr_path, train_arr)
            np.save(self.data_transformation_config.test_arr_path, test_arr)

            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj = DataTransformation()
    # We pass the paths directly from what Ingestion generated
    train_array, test_array, preprocessor_path = obj.initiate_data_transformation(
        'artifacts/train.csv', 
        'artifacts/test.csv'
    )
    print(f"Transformation complete. Preprocessor saved at: {preprocessor_path}")