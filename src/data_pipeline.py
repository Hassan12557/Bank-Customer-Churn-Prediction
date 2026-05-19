"""
Automated Customer Churn Data Pipeline
Author: Data Science Team
Description: Automates end-to-end data processing, handling missing values,
             categorical encoding, and time-based splits defensively.
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ==========================================
# 1. CUSTOM CUSTOMER FEATURE TRANSFORMERS
# ==========================================

class DemographicsInteracter(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineered composite interactions safely
    without generating out-of-bounds errors on unseen production strings.
    """

    def __init__(self, gender_col='Gender', marital_col='MaritalStatus'):
        self.gender_col = gender_col
        self.marital_col = marital_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        if self.gender_col in X_out.columns and self.marital_col in X_out.columns:
            # Enforce string conversion to block NaN concatenation errors
            X_out['Gender_Marital'] = (X_out[self.gender_col].astype(str) + '_' +
                                       X_out[self.marital_col].astype(str))
        return X_out


# ==========================================
# 2. MAIN AUTOMATED DATA PIPELINE ENGINE
# ==========================================

class ChurnDataPipeline:
    def __init__(self, raw_data_path="data/raw_customer_data.csv",
                 processed_dir="data/processed"):
        self.raw_data_path = raw_data_path
        self.processed_dir = processed_dir
        self.preprocessor = None

        # Define internal column tracking manifests
        self.numeric_features = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_features = ['Gender_Marital', 'ResolutionStatus', 'InteractionType', 'IncomeLevel']

    def load_raw_data(self):
        """Loads and chronologically stabilizes the dataframe matrix."""
        if not os.path.exists(self.raw_data_path):
            print(f"⚠️ Raw data file '{self.raw_data_path}' not found! Creating clean mock simulation file...")
            self._generate_mock_raw_data()

        df = pd.read_csv(self.raw_data_path)

        # Standardize chronological ordering to block time-travel lookahead leakage
        if 'TransactionDate' in df.columns:
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
            df = df.sort_values('TransactionDate').reset_index(drop=True)
            print("✓ Chronological order verified and enforced across data indices.")

        return df

    def build_processing_pipeline(self):
        """Defines structural boundaries for mathematical transformations."""

        # Numeric pipeline: Impute missing entries with median, then standardize scales
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline: Handle missing strings, then apply One-Hot encoding safely
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine tracking components under a unified ColumnTransformer layout
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Safely drop non-engineered metadata columns
        )

        # Build global structural pipeline
        full_pipeline = Pipeline(steps=[
            ('interactions', DemographicsInteracter()),
            ('preprocessor', self.preprocessor)
        ])

        return full_pipeline

    def run_automation(self):
        """Executes full automated pipeline sequence and saves production artifacts."""
        print("\n" + "=" * 60)
        print("          STARTING AUTOMATED DATA PIPELINE")
        print("=" * 60)

        df = self.load_raw_data()

        if 'ChurnStatus' not in df.columns:
            raise KeyError("Critical Failure: Target tracker column 'ChurnStatus' is missing from file.")

        # Time-Based Split Execution (Preserves strict validation logic)
        split_idx = int(0.8 * len(df))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        y_train = train_df['ChurnStatus'].values
        y_test = test_df['ChurnStatus'].values

        # Construct and fit global transform protocols strictly on training vectors
        pipeline_model = self.build_processing_pipeline()

        print("Transforming and aligning data splits...")
        X_train_transformed = pipeline_model.fit_transform(train_df)
        X_test_transformed = pipeline_model.transform(test_df)
   

	# === ADD THIS BLOCK TO SAVE YOUR PREPROCESSOR ===
        import joblib
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(pipeline_model, os.path.join(models_dir, "preprocessor_pipeline.pkl"))
        print("✓ Preprocessing pipeline architecture saved to 'models/preprocessor_pipeline.pkl'")

        # Dynamically map and rebuild exact column name arrays for tracking transparency
        cat_encoder = pipeline_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        encoded_cat_cols = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
        all_features = self.numeric_features + encoded_cat_cols

        # Re-assemble data matrices into clean DataFrames
        train_clean = pd.DataFrame(X_train_transformed, columns=all_features)
        train_clean['ChurnStatus'] = y_train

        test_clean = pd.DataFrame(X_test_transformed, columns=all_features)
        test_clean['ChurnStatus'] = y_test

        # Save processed outputs to disk
        os.makedirs(self.processed_dir, exist_ok=True)
        train_clean.to_csv(os.path.join(self.processed_dir, "train_processed.csv"), index=False)
        test_clean.to_csv(os.path.join(self.processed_dir, "test_processed.csv"), index=False)

        print(f"✓ Pipeline execution complete!")
        print(f"  - Training array shape: {train_clean.shape} saved to '{self.processed_dir}/train_processed.csv'")
        print(f"  - Testing array shape:  {test_clean.shape} saved to '{self.processed_dir}/test_processed.csv'")
        print("=" * 60 + "\n")

    def _generate_mock_raw_data(self):
        """Generates structured mock raw data if standard file paths aren't found."""
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        np.random.seed(42)
        n_samples = 1200

        mock_df = pd.DataFrame({
            'TransactionDate': pd.date_range(start='2025-01-01', periods=n_samples, freq='h'),
            'Age': np.random.randint(18, 75, n_samples),
            'Tenure': np.random.randint(1, 72, n_samples),
            'MonthlyCharges': np.random.uniform(29.99, 120.00, n_samples),
            'TotalCharges': np.random.uniform(100.00, 5000.00, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], size=n_samples),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], size=n_samples),
            'ResolutionStatus': np.random.choice(['Resolved', 'Unresolved', 'Escalated'], size=n_samples),
            'InteractionType': np.random.choice(['Call Center', 'Web Chat', 'Email'], size=n_samples),
            'IncomeLevel': np.random.choice(['Low', 'Medium', 'High'], size=n_samples),
            'ChurnStatus': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        })
        # Inject intentional NaNs to test pipeline stability
        mock_df.loc[mock_df.sample(frac=0.05).index, 'TotalCharges'] = np.nan
        mock_df.to_csv(self.raw_data_path, index=False)
        print(f"✓ Generated automated fallback file containing {n_samples} simulated user accounts.")


if __name__ == '__main__':
    # Initialize and execute pipeline automation setup
    pipeline_manager = ChurnDataPipeline(
        raw_data_path="data/raw_customer_data.csv",
        processed_dir="data/processed"
    )
    pipeline_manager.run_automation()