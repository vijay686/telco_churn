# Telco Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecommunications customer is likely to churn.

Customer churn prediction helps telecom companies identify customers who may leave their service. By detecting high-risk customers early, businesses can take preventive actions such as offering personalized plans, discounts, or improved support.

---

## Project Overview

This project implements a complete machine learning pipeline for customer churn prediction.

The pipeline includes:

- Data ingestion
- Data validation
- Data transformation
- Model training
- Model evaluation
- Model prediction
- Logging
- Custom exception handling
- Configuration-based pipeline execution
- Application-based prediction

The project is organized using a modular structure so that each stage can be developed, tested, and maintained independently.

---

## Problem Statement

The objective of this project is to predict customer churn using customer demographic, account, and service-related information.

The target variable indicates whether a customer has churned:

- `Yes` — The customer has churned.
- `No` — The customer has not churned.

This is a binary classification problem.

---

## Features

The project may use customer attributes such as:

- Gender
- Senior citizen status
- Partner and dependent information
- Tenure
- Phone and internet services
- Contract type
- Payment method
- Monthly charges
- Total charges
- Online security and support services
- Customer churn status

The exact features depend on the dataset used during training and prediction.

---

## Machine Learning Workflow

```text
Raw Dataset
     |
     v
Data Ingestion
     |
     v
Data Validation
     |
     v
Data Transformation
     |
     v
Model Training
     |
     v
Model Evaluation
     |
     v
Trained Model
     |
     v
Customer Churn Prediction
```

---

## Project Structure

```text
telco_churn/
│
├── config/
│   └── config.yaml
│
├── notebook/
│   └── data/
│       └── Dataset and notebook-related files
│
├── src/
│   └── telco_churn/
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       │
│       ├── configuration/
│       │   └── configuration.py
│       │
│       ├── constants/
│       │   └── __init__.py
│       │
│       ├── entity/
│       │   └── config_entity.py
│       │
│       ├── pipeline/
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_validation.py
│       │   ├── stage_03_data_transformation.py
│       │   ├── stage_04_model_trainer.py
│       │   └── stage_05_model_evaluation.py
│       │
│       ├── utils/
│       │   └── common.py
│       │
│       ├── logger.py
│       └── exception.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── artifacts/
│
├── app.py
├── predict.py
├── config.yaml
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

> The exact names of files inside `src/` may vary depending on the current implementation. Keep this structure synchronized with the actual repository whenever files are added or renamed.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyYAML
- Joblib or Pickle
- Flask or Streamlit, depending on the application implementation
- Logging
- Custom exception handling
- Setuptools
- Git and GitHub

---

## Installation

### 1. Clone the repository

```bash
git clone [https://github.com/vijay686/telco_churn.git](https://github.com/vijay686/telco_churn.git)
cd telco_churn
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux or macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available or is incomplete, install the main dependencies manually:

```bash
pip install pandas numpy scikit-learn pyyaml joblib flask streamlit
```

### 4. Install the project as a package

```bash
pip install -e .
```

The `-e` option installs the project in editable mode, which means changes made to the source code are immediately available without reinstalling the package.

---

## Configuration

The project uses YAML configuration files to manage paths and pipeline settings.

A typical configuration file may contain:

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: <dataset-url>
  local_data_file: artifacts/data_ingestion/data.csv
  unzip_dir: artifacts/data_ingestion
```

Before running the pipeline, verify that:

- Dataset paths are correct.
- Artifact directories exist or can be created.
- The dataset URL is valid if downloading data automatically.
- Model and output paths match the application code.

Do not store passwords, API keys, or other sensitive information in configuration files committed to GitHub.

---

## Running the Project

### Run the complete pipeline

If the project provides a main pipeline entry point, run:

```bash
python main.py
```

If individual pipeline stages are provided, run them in order:

```bash
python src/telco_churn/pipeline/stage_01_data_ingestion.py
python src/telco_churn/pipeline/stage_02_data_validation.py
python src/telco_churn/pipeline/stage_03_data_transformation.py
python src/telco_churn/pipeline/stage_04_model_trainer.py
python src/telco_churn/pipeline/stage_05_model_evaluation.py
```

Run only the commands that correspond to files present in the repository.

---

## Making Predictions

The `predict.py` file is used to generate predictions using the trained model.

Run:

```bash
python predict.py
```

A typical prediction workflow is:

1. Load the trained model.
2. Load the preprocessing object.
3. Read customer input data.
4. Apply the same preprocessing used during training.
5. Generate a churn prediction.
6. Display or return the prediction result.

Example output:

```text
Prediction: Customer is likely to churn
```

The prediction input must contain the same feature names and compatible data types used during model training.

---

## Model Evaluation

The model should be evaluated using appropriate classification metrics, such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

For churn prediction, accuracy alone may not be sufficient. Precision and recall are especially important because:

- False negatives may represent customers who are predicted as safe but actually churn.
- False positives may lead to unnecessary retention offers.

The final model should be selected based on business requirements and not only on the highest accuracy score.

---

## Logging and Exception Handling

The project includes logging and custom exception handling to make errors easier to identify.

Logs can help track:

- Pipeline stage execution
- Dataset loading
- Validation results
- Model training progress
- Evaluation results
- Runtime errors

When an error occurs, check the generated log files before debugging individual components.

---

## Output Artifacts

The pipeline may generate artifacts such as:

```text
artifacts/
├── data_ingestion/
├── data_validation/
├── data_transformation/
├── model_trainer/
│   └── trained_model.pkl
└── model_evaluation/
    └── metrics.json
```

Generated datasets, trained models, logs, and temporary files should generally not be committed to GitHub unless they are intentionally included for demonstration.

---

## Reproducibility

To reproduce the project:

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the dependencies.
4. Verify the configuration file.
5. Add or download the dataset.
6. Run the pipeline stages in order.
7. Run the prediction script or application.

Use fixed random seeds wherever possible so that training results can be reproduced.

---


## Author

**Vijay**

GitHub: [@vijay686](https://github.com/vijay686)

Repository: [telco_churn](https://github.com/vijay686/telco_churn)

---

