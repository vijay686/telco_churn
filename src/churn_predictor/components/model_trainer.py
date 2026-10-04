import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.churn_predictor.exception import CustomException
from src.churn_predictor.logger import logging
from src.churn_predictor.utils import save_object
# from src.churn_predictor.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting transformed data into dependent and independent features")
            # Assuming the Machining Floor placed the target variable 'y' in the last column
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # The Arena: Instantiate the combatants
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced"),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
            }

            model_report = {}

            logging.info("Commencing model training and evaluation sequence...")
            
            for name, model in models.items():
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate combat metrics
                f1 = f1_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                
                model_report[name] = f1
                logging.info(f"{name} -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

            # Extract the champion based on highest F1-Score
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.55:
                raise CustomException("Critical Failure: No algorithm achieved an acceptable F1-Score.", sys)

            logging.info(f"Champion Crowned: {best_model_name} (F1-Score: {best_model_score})")

            # Save the champion to the artifacts directory
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)