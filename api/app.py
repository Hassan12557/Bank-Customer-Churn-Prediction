"""
Bank Customer Churn Prediction GUI Dashboard
Author: Data Science Team
Description: An interactive, production-grade Streamlit web interface
             engineered with strict compliance to HCI principles.
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# --- 1. SET PAGE CONFIGURATION & THEME AMBIENCE ---
st.set_page_config(
    page_title="RiskHorizon™ | Churn Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Corporate Layout Styling (HCI: Aesthetic-Usability Effect)
st.markdown("""
    <style>
        /* Main structural card formatting */
        .metric-card {
            background-color: #1e293b;
            padding: 24px;
            border-radius: 12px;
            border-left: 5px solid #3b82f6;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            margin-bottom: 20px;
        }
        .metric-card.danger { border-left-color: #ef4444; }
        .metric-card.warning { border-left-color: #f59e0b; }
        .metric-card.success { border-left-color: #10b981; }

        /* Typography overrides */
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #3b82f6, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)  # <<--- FIXED HERE
# --- 2. CONFIGURING SYSTEM PATHS & DEPENDENCIES ---
API_DIR = Path(__file__).resolve().parent
BASE_DIR = API_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Fallback path checking for local src development layouts
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from data_pipeline import DemographicsInteracter
except ImportError:
    st.error("❌ Critical System Error: 'data_pipeline.py' not found.")
    st.stop()


# --- 3. CACHING SERIALIZED ML ARTIFACTS ---
@st.cache_resource
def load_ml_assets():
    pipeline_path = BASE_DIR / "models" / "preprocessor_pipeline.pkl"
    model_path = BASE_DIR / "models" / "best_churn_model.pkl"

    if not pipeline_path.exists() or not model_path.exists():
        return None, None

    pipeline_obj = joblib.load(pipeline_path)
    model_obj = joblib.load(model_path)
    return pipeline_obj, model_obj


pipeline, model = load_ml_assets()

if pipeline is None or model is None:
    st.error("⚠️ Model Artifacts Missing! Please run pipelines to generate `.pkl` files.")
    st.stop()
    # --- 4. GRAPHICAL HEADER ---
st.markdown('<p class="main-title">RiskHorizon™ Customer Analytics</p>', unsafe_allow_html=True)  # <<--- FIXED HERE
st.markdown('<p class="subtitle">Enterprise Real-Time Risk Profiling & Churn Mitigation Console</p>',
            unsafe_allow_html=True)  # <<--- FIXED HERE

# --- 5. HCI LAYOUT: STEPPED DATA INPUT CHUNKING (Miller's Law) ---
# Instead of a confusing wall of variables, we separate forms by context
st.markdown("### 📝 Enter Profile Parameters")
tab1, tab2, tab3 = st.tabs(["👤 Core Demographics", "💰 Financial Footprint", "⚡ Account Interactions"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age of Account Holder", min_value=18, max_value=100, value=35)
        gender = st.radio("Gender Profile", ["Male", "Female"], horizontal=True)
    with col2:
        marital_status = st.selectbox("Marital Class Matrix", ["Single", "Married", "Divorced"])
        st.markdown(
            "<br><p style='color: #64748b;'>Note: Demographics are combined dynamically into custom interaction arrays downstream.</p>",
            unsafe_output_html=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Account Lifespan (Months Active)", min_value=1, max_value=72, value=24)
        monthly_charges = st.number_input("Current Monthly Statement Valuation ($)", min_value=10.0, max_value=300.0,
                                          value=65.0, step=5.0)
    with col2:
        total_charges = st.number_input("Historical Aggregated Billings ($)", min_value=10.0, max_value=50000.0,
                                        value=1500.0, step=100.0)
        income_level = st.select_slider("Assigned Income Tier", options=["Low", "Medium", "High"], value="Medium")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        resolution_status = st.selectbox("Latest Service Complaint Status", ["Resolved", "Unresolved", "Escalated"])
        interaction_type = st.selectbox("Primary Engagement Touchpoint", ["Web Chat", "Call Center", "Email"])
    with col2:
        st.markdown("""
            <div style="background-color: #0f172a; padding: 16px; border-radius: 8px; border: 1px solid #334155;">
                <p style="margin: 0; font-weight: bold; color: #3b82f6;">⚠️ Operational Impact Mapping</p>
                <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #94a3b8;">
                    Unresolved and Escalated tickets carry massive relative feature weight correlations according to Random Forest and XGBoost profiling modules.
                </p>
            </div>
        """, unsafe_output_html=True)

st.markdown("<br>", unsafe_output_html=True)
predict_button = st.button("🚀 Execute Strategic Risk Assessment", type="primary", use_container_width=True)
st.markdown("---")

# --- 6. REAL-TIME PREDICTION & VISUALIZATION ENGINE ---
if predict_button:
    # Build Input Frame
    raw_input_df = pd.DataFrame({
        'Age': [age], 'Tenure': [tenure], 'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges],
        'Gender': [gender], 'MaritalStatus': [marital_status], 'ResolutionStatus': [resolution_status],
        'InteractionType': [interaction_type], 'IncomeLevel': [income_level]
    })

    # Inform the user with an active UI loading state (HCI: Continuous Visibility)
    with st.spinner("Processing customer features through automated data pipeline..."):
        try:
            processed_vector = pipeline.transform(raw_input_df)
            risk_probability = model.predict_proba(processed_vector)[0, 1]
        except Exception as e:
            st.error(f"💥 Transformation Pipeline Failure: {e}")
            st.stop()

    st.markdown("### 🎯 Strategic Assessment Insights")

    # Dynamic styling values based on risk thresholds
    if risk_probability > 0.50:
        card_class, alert_type, title, txt = "danger", "error", "⚠️ HIGH CHURN RISK CRITICAL VULNERABILITY", "Customer characteristics strongly align with historical churn vectors. Immediate mitigation campaign strategies must be activated."
        gauge_color = "#ef4444"
    elif risk_probability > 0.20:
        card_class, alert_type, title, txt = "warning", "warning", "🟡 MODERATE TRANSITIONAL RISK PROFILE", "Account displays shifting behavior flags. Monitor communication channels closely for additional friction points."
        gauge_color = "#f59e0b"
    else:
        card_class, alert_type, title, txt = "success", "success", "✅ STRUCTURALLY SECURE / HIGH RETENTION", "Operational metrics reflect highly resilient customer patterns. Maintain existing systemic relationship workflows."
        gauge_color = "#10b981"

    # Display split layout: Dial gauge chart on left, concrete corporate summary on right
    g_col, c_col = st.columns([2, 3])

    with g_col:
        # High-Fidelity Donut Gauge (HCI: Immediate Visual Cognition)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 44, 'family': 'sans-serif', 'color': '#ffffff'}},
            title={'text': "Calculated Attrition Vector", 'font': {'size': 16, 'color': '#94a3b8'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': gauge_color},
                'bgcolor': "#1e293b",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.05)'},
                    {'range': [20, 50], 'color': 'rgba(245, 158, 11, 0.05)'},
                    {'range': [50, 100], 'color': 'rgba(239, 68, 68, 0.05)'}
                ],
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10),
            height=250
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with c_col:
        # HTML Rendered Status Card
        st.markdown(f"""
                    <div class="metric-card {card_class}">
                        <h4 style="margin: 0 0 8px 0; color: #ffffff; font-weight: bold;">{title}</h4>
                        <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">{txt}</p>
                    </div>
                """, unsafe_allow_html=True)  # <<--- FIXED HERE
        # Financial Yield Impact Callout Matrix
        retention_value = 500
        intervention_cost = 50
        expected_margin = (retention_value * (1 - risk_probability)) - intervention_cost

        m1, m2 = st.columns(2)
        m1.metric("Mitigation Allocation Cost", f"${intervention_cost}.00",
                  help="Standard pricing budget allocated for outbound customer recovery scripts.")
        m2.metric("Target Optimization Yield", f"${expected_margin:,.2f}",
                  help="Calculated net utility return mapping of customer account lifetime valuation adjustments.")

    # Technical audit logs hidden safely below fold (HCI: Progressive Disclosure)
    with st.expander("🛠️ Advanced Architectural Diagnostics & Audit Rails"):
        st.json({
            "model_type": str(type(model)),
            "pipeline_steps": list(pipeline.named_steps.keys()),
            "input_dimensions": raw_input_df.shape,
            "raw_features_evaluated": list(raw_input_df.columns),
            "inference_confidence_interval": float(np.max(model.predict_proba(processed_vector)[0]))
        })