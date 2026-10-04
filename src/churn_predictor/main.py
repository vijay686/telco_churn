import sys
from src.churn_predictor.logger import logging
from src.churn_predictor.exception import CustomException

# Importing the components we built in Phases 3, 5, and 6
from src.churn_predictor.components.data_ingestion import DataIngestion
from src.churn_predictor.components.data_transformation import DataTransformation
from src.churn_predictor.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info(">>> Initializing Training Pipeline Execution <<<")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()
        
        # Step 2: Data Transformation
        transformation = DataTransformation()
        train_array, test_array, preprocessor_path = transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        
        # Step 3: Model Training
        trainer = ModelTrainer()
        f1_score = trainer.initiate_model_trainer(train_array, test_array)
        
        logging.info(f"Pipeline execution completed successfully. Champion Model F1 Score: {f1_score}")

    except Exception as e:
        logging.error("Pipeline execution terminated due to an error.")
        raise CustomException(e, sys)