import os
import sys
import pandas as pd
import numpy as np

from src.churn_predictor.exception import CustomException
from src.churn_predictor.logger import logging
from src.churn_predictor.utils import load_object


class BatchPredictionPipeline:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.model_path = os.path.join(artifacts_dir, "model.pkl")
        self.preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")

    def run_batch_inference(self, input_csv_path: str, output_csv_path: str = "artifacts/batch_predictions.csv"):
        try:
            logging.info(f"Loading batch dataset from: {input_csv_path}")
            if not os.path.exists(input_csv_path):
                raise FileNotFoundError(f"Input file not found at: {input_csv_path}")

            raw_df = pd.read_csv(input_csv_path)
            logging.info(f"Ingested {len(raw_df)} records for batch scoring.")

            # Preserve identifiers if present for the final deliverable
            customer_ids = raw_df["customerID"] if "customerID" in raw_df.columns else pd.Series(range(len(raw_df)))

            # Step 1: Sanitize Features
            df = raw_df.copy()

            # Handle the TotalCharges whitespace trap
            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
                # The Modern, CoW-Compliant Standard
                df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

            # Drop non-feature columns (ID and Target if present)
            drop_cols = [col for col in ["customerID", "Churn"] if col in df.columns]
            features_df = df.drop(columns=drop_cols)

            # Step 2: Load Frozen Artifacts
            logging.info("Deserializing model and preprocessor artifacts.")
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            # Step 3: Transformation & Model Inference
            logging.info("Executing preprocessor.transform on feature batch.")
            transformed_features = preprocessor.transform(features_df)

            logging.info("Calculating predictions and calibrated probabilities.")
            predictions = model.predict(transformed_features)
            
            # Extract probability of Churn (Class 1) if supported by the model
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed_features)[:, 1]
            else:
                probabilities = np.nan

            # Step 4: Assemble the Business Deliverable
            output_df = raw_df.copy()
            output_df["Predicted_Churn_Label"] = np.where(predictions == 1, "Yes", "No")
            output_df["Churn_Risk_Score"] = np.round(probabilities, 4)

            # Ensure output directory exists and save
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            output_df.to_csv(output_csv_path, index=False)

            logging.info(f"Batch prediction successfully generated and saved to: {output_csv_path}")
            print(f"\n[SUCCESS] Batch predictions exported to: {output_csv_path}")
            print(output_df[["customerID", "Predicted_Churn_Label", "Churn_Risk_Score"]].head() if "customerID" in output_df.columns else output_df[["Predicted_Churn_Label", "Churn_Risk_Score"]].head())

            return output_df

        except Exception as e:
            logging.error("Batch inference pipeline encountered a critical failure.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test run against your test dataset or raw dataset
    pipeline = BatchPredictionPipeline()
    
    # Point this to whichever CSV you are evaluating
    target_csv = os.path.join("artifacts", "test.csv")
    
    if os.path.exists(target_csv):
        pipeline.run_batch_inference(input_csv_path=target_csv)
    else:
        print(f"Target test CSV not found at {target_csv}. Provide a valid CSV path.")