"""
Production Inference Script
Description: Loads saved pipeline and model artifacts to predict on brand new customer data.
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd

# 1. Dynamically locate the project root directory and add to system path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# === FIXED: Import your custom transformer from data_pipeline.py ===
# This tells joblib exactly what 'DemographicsInteracter' is when unpickling!
from data_pipeline import DemographicsInteracter

MODELS_DIR = BASE_DIR / "models"
pipeline_path = MODELS_DIR / "preprocessor_pipeline.pkl"
model_path = MODELS_DIR / "best_churn_model.pkl"

# Double check that the files actually exist before attempting to load them
if not pipeline_path.exists() or not model_path.exists():
    raise FileNotFoundError(
        f"Could not find model artifacts at '{MODELS_DIR}'.\n"
        f"Please run 'data_pipeline.py' followed by 'train_model.py' first to generate them!"
    )

# Load the saved pipeline and the champion model using absolute paths
pipeline = joblib.load(pipeline_path)
model = joblib.load(model_path)
print("✓ Model and preprocessing pipeline successfully loaded into memory from project root.")

# 2. Simulate incoming live production data (Unseen by the model)
new_customers = pd.DataFrame({
    'Age': [34, 52],
    'Tenure': [12, 4],
    'MonthlyCharges': [89.50, 45.00],
    'TotalCharges': [1074.00, 180.00],
    'Gender': ['Female', 'Male'],
    'MaritalStatus': ['Single', 'Married'],
    'ResolutionStatus': ['Resolved', 'Escalated'],
    'InteractionType': ['Web Chat', 'Call Center'],
    'IncomeLevel': ['Medium', 'Low']
})

# 3. Transform the data using the exact same pipeline parameters
transformed_data = pipeline.transform(new_customers)

# 4. Generate Predictions
predictions = model.predict(transformed_data)
probabilities = model.predict_proba(transformed_data)[:, 1]

# 5. Output results
print("\n--- Inference Results ---")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    status = "Likely to Churn" if pred == 1 else "Safe / Active"
    print(f"Customer {i+1}: Status -> {status} (Risk Probability: {prob:.2%})")