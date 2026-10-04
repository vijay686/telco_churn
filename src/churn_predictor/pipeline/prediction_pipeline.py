import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Any
import pandas as pd

from src.churn_predictor.exception import CustomException
from src.churn_predictor.logger import logging
from src.churn_predictor.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        """
        Loads frozen artifacts, applies feature transformations, and outputs inferences.
        """
        try:
            logging.info("Starting inference run.")

            if not (os.path.exists(self.model_path) and os.path.exists(self.preprocessor_path)):
                raise FileNotFoundError("Model or preprocessor artifact not found in artifacts directory.")

            # Load serialized pipeline objects
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            logging.info("Transforming incoming feature DataFrame.")
            scaled_features = preprocessor.transform(features)

            logging.info("Executing model inference.")
            predictions = model.predict(scaled_features)
            
            return predictions

        except Exception as e:
            logging.error("Inference step failed.")
            raise CustomException(e, sys)


@dataclass(frozen=True)
class CustomerDataContract:
    """
    Production schema specification for Telco customer churn.
    Encapsulates all 19 raw inference features with type guarantees.
    """
    # Demographics
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    
    # Account & Billing Metrics
    tenure: int
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
    # Phone Services
    PhoneService: str
    MultipleLines: str
    
    # Internet Services & Add-ons
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataclass record into a single-row Pandas DataFrame 
        with exact column casing required by the ColumnTransformer.
        """
        try:
            record_dict = {k: [v] for k, v in asdict(self).items()}
            df = pd.DataFrame(record_dict)
            return df
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Factory interface to ingest form data, web queries, or JSON payloads
    and instantiate the validated CustomerDataContract.
    """
    def __init__(self, **kwargs: Any):
        self.features = kwargs

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            # Instantiate dataclass contract to validate presence and map schema
            contract = CustomerDataContract(**self.features)
            return contract.to_dataframe()
        except TypeError as type_err:
            logging.error(f"Payload schema validation error: {type_err}")
            raise CustomException(type_err, sys)
        except Exception as e:
            raise CustomException(e, sys)