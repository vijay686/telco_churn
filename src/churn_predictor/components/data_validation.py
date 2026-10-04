import os
import sys
import pandas as pd
import yaml
from dataclasses import dataclass
from src.churn_predictor.logger import logging
from src.churn_predictor.exception import CustomException

# The Configuration Blueprint
@dataclass
class DataValidationConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    schema_path: str = 'schema.yaml'
    status_file_path: str = os.path.join('artifacts', 'status.txt')

# The Component Logic
class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def validate_all_columns(self)-> bool:
        try:
            validation_status = None
            
            # Read the incoming data and the rulebook
            train_data = pd.read_csv(self.validation_config.train_data_path)
            
            with open(self.validation_config.schema_path, "r") as file:
                schema = yaml.safe_load(file)
            
            schema_columns = list(schema['COLUMNS'].keys())
            data_columns = list(train_data.columns)
            
            # Schema Check
            for col in data_columns:
                if col not in schema_columns:
                    validation_status = False
                    with open(self.validation_config.status_file_path, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    break
                else:
                    validation_status = True
                    with open(self.validation_config.status_file_path, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            logging.info(f"Data Validation completed. Status: {validation_status}")
            return validation_status

        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":
    try:
        obj = DataValidation()
        status = obj.validate_all_columns()
        print(f"Validation finished. Pipeline safe to proceed: {status}")
    except Exception as e:
        print(e)