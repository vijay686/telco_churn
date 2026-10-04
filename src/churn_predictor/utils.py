import os
import sys
import pickle
from src.churn_predictor.exception import CustomException
from src.churn_predictor.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Loads and returns a Python object from a pickle file.
    """

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Pickle file not found at path: {file_path}"
            )

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)