"""
Bank Customer Churn Prediction GUI Dashboard
Author: Data Science Team
Description: An interactive, production-grade Streamlit web interface
             for real-time individual customer churn risk profiling.
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# --- 1. SET PAGE CONFIGURATION & THEME AMBIENCE ---
st.set_page_config(
    page_title="Customer Churn Prediction Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CONFIGURING SYSTEM PATHS & DEPENDENCIES ---
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Import your custom transformer so joblib knows how to unpickle the preprocessor
try:
    from data_pipeline import DemographicsInteracter
except ImportError:
    st.error("❌ Critical System Error: 'data_pipeline.py' not found in project path.")
    st.stop()


# --- 3. CACHING SERIALIZED ML ARTIFACTS ---
@st.cache_resource
def load_ml_assets():
    """Loads and caches models to avoid re-reading disk on every user click."""
    pipeline_path = BASE_DIR / "models" / "preprocessor_pipeline.pkl"
    model_path = BASE_DIR / "models" / "best_churn_model.pkl"

    if not pipeline_path.exists() or not model_path.exists():
        return None, None

    pipeline_obj = joblib.load(pipeline_path)
    model_obj = joblib.load(model_path)
    return pipeline_obj, model_obj


pipeline, model = load_ml_assets()

# Safeguard check if pipeline files don't exist yet
if pipeline is None or model is None:
    st.error("⚠️ Model Artifacts Missing!")
    st.info(
        "Please run `data_pipeline.py` and then `train_model.py` to generate the saved pickle (.pkl) files inside the `models/` directory.")
    st.stop()

# --- 4. GRAPHICAL USER INTERFACE HEADER ---
st.title("📊 Customer Churn Risk Prediction Terminal")
st.markdown("""
This production interface utilizes your trained machine learning champion architecture to predict customer attrition risks in real time. 
Input the customer attributes below to evaluate metrics.
""")
st.write("---")

# --- 5. RESPONSIVE MULTI-COLUMN DATA INPUT FORM ---
st.subheader("👤 Customer Profile Matrix")

# Divide form into columns for a clean look
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age of Account Holder", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

with col2:
    tenure = st.slider("Tenure (Months Active)", min_value=1, max_value=120, value=24)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=500.0, value=75.0, step=0.50)
    total_charges = st.number_input("Total Historical Charges ($)", min_value=10.0, max_value=50000.0, value=1800.0,
                                    step=10.0)

with col3:
    resolution_status = st.selectbox("Latest Ticket Resolution Status", ["Resolved", "Unresolved", "Escalated"])
    interaction_type = st.selectbox("Primary Communication Channel", ["Web Chat", "Call Center", "Email"])
    income_level = st.selectbox("Estimated Income Bracket", ["Low", "Medium", "High"])

st.write("")
predict_button = st.button("🚀 Analyze Churn Risk Profile", type="primary", use_container_width=True)
st.write("---")

# --- 6. REAL-TIME PREDICTION & INFERENCE ENGINE ---
if predict_button:
    # 1. Structure raw inputs exactly to match original DataFrame features
    raw_input_df = pd.DataFrame({
        'Age': [age],
        'Tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Gender': [gender],
        'MaritalStatus': [marital_status],
        'ResolutionStatus': [resolution_status],
        'InteractionType': [interaction_type],
        'IncomeLevel': [income_level]
    })

    # 2. Process data through the automated pipeline
    with st.spinner("Processing customer features through data pipeline..."):
        try:
            processed_vector = pipeline.transform(raw_input_df)

            # Generate inference scores
            risk_probability = model.predict_proba(processed_vector)[0, 1]
            predicted_class = model.predict(processed_vector)[0]

        except Exception as e:
            st.error(f"💥 Transformation Pipeline Failure: {e}")
            st.stop()

    # 3. Render Metric Cards and Risk Alerts Dynamically
    st.subheader("🎯 Risk Assessment Output")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(label="Risk Probability Score", value=f"{risk_probability:.2%}")

    with m_col2:
        if risk_probability > 0.50:
            st.metric(label="System Class Verdict", value="⚠️ HIGH RISK (Churn Likely)")
        else:
            st.metric(label="System Class Verdict", value="✅ SAFE (Retention High)")

    # Display clear warning cards depending on status thresholds
    if risk_probability > 0.50:
        st.error(
            f"🔴 **Critical Alert:** This customer exhibits severe indicators of churn vector alignment. Recommended strategic intervention action value threshold: `$50.00` outreach optimization.")
    elif risk_probability > 0.20:
        st.warning(
            f"🟡 **Notice:** Customer is in a moderate risk transition bracket. Recommend proactive monitoring through the **{interaction_type}** support pipeline.")
    else:
        st.success(
            f"🟢 **Optimal Performance:** Customer is structurally stable. Operational account footprint indicates long-term active usage pattern continuity.")

    # Expandable details showing transformed array inputs for audit transparency
    with st.expander("Technical Diagnostic Array Details"):
        st.write("Raw User Entry Frame:")
        st.dataframe(raw_input_df)
        st.write(f"Post-Pipeline Vector Matrix Representation Shape: {processed_vector.shape}")