"""
Bank Customer Churn Prediction GUI Dashboard
Style Archetype: React.js Glassmorphic / Fintech Premium Dark Theme
Author: Advanced AI Systems Team
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests

# --- 1. APP CONFIGURATION & REACT AMBIENCE ---
st.set_page_config(
    page_title="RiskHorizon™ OS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define network routing boundary to the backend Docker service
API_URL = "http://backend:8000/predict"

# Premium React-Style CSS Injection Framework (With Extended Responsive Properties)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #060913 !important;
            color: #f1f5f9;
        }

        /* Premium Glassmorphic Layout Base */
        .react-card {
            background: rgba(15, 23, 42, 0.45);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 20px;
            padding: 28px;
            margin-bottom: 24px;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .react-card:hover {
            border: 1px solid rgba(59, 130, 246, 0.25);
            box-shadow: 0 30px 60px -15px rgba(59, 130, 246, 0.12);
            transform: translateY(-2px);
        }

        /* Typography & Header Enhancements */
        .react-hero-title {
            font-size: clamp(2rem, 4vw, 3.2rem);
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.1;
            background: linear-gradient(135deg, #93c5fd 0%, #3b82f6 50%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }
        .react-hero-sub {
            color: #475569;
            font-size: clamp(0.9rem, 1.5vw, 1.1rem);
            font-weight: 500;
            margin-bottom: 32px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Interactive CRM Tasks Formats */
        .action-header {
            font-weight: 700;
            color: #f8fafc;
            font-size: 1.15rem;
            margin-bottom: 16px;
            letter-spacing: -0.01em;
        }
        .task-item {
            background: rgba(30, 41, 59, 0.3);
            padding: 14px 18px;
            border-radius: 12px;
            border-left: 4px solid #475569;
            margin-bottom: 10px;
            font-size: 0.92rem;
            color: #cbd5e1;
            transition: transform 0.2s ease;
        }
        .task-item:hover {
            transform: translateX(4px);
        }
        .task-item.high { border-left-color: #ef4444; background: rgba(239, 68, 68, 0.03); }
        .task-item.medium { border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.03); }
        .task-item.low { border-left-color: #10b981; background: rgba(16, 185, 129, 0.03); }

        /* Clean Streamlit Overrides for Full Responsiveness */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
        }
        [data-testid="stExpander"] {
            background: rgba(15, 23, 42, 0.2) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 16px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. FINTECH HERO DISPLAY SECTION ---
st.markdown('<h1 class="react-hero-title">RiskHorizon™ Intelligence OS</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="react-hero-sub"><span style="color:#3b82f6;">●</span> Next-Gen Customer Attrition Engine • Cloud Environment Active v4.0.0</p>',
    unsafe_allow_html=True)

# --- 3. STATE-DRIVEN ACCOUNT PARAMETERS INPUT ---
# Using an interactive expander as a slide-out drawer pattern to manage screen real estate responsively
with st.expander("📥 Adjust Real-Time Customer Account Telemetry Input Profile", expanded=True):
    st.markdown("<br>", unsafe_allow_html=True)

    # 4-Column flexible layout structure
    c1, c2, c3, c4 = st.columns([1, 1.1, 1.1, 1])

    with c1:
        st.markdown(
            "<p style='color:#64748b; font-size:0.75rem; font-weight:700; letter-spacing:0.05em; margin-bottom:12px;'>DEMOGRAPHICS</p>",
            unsafe_allow_html=True)
        age = st.number_input("Age Profile", min_value=18, max_value=95, value=42)
        gender = st.selectbox("Gender Orientation", ["Male", "Female"])
        marital_status = st.selectbox("Marital Allocation", ["Single", "Married", "Divorced"])

    with c2:
        st.markdown(
            "<p style='color:#64748b; font-size:0.75rem; font-weight:700; letter-spacing:0.05em; margin-bottom:12px;'>FINANCIAL STRUCTURE</p>",
            unsafe_allow_html=True)
        monthly_charges = st.number_input("Monthly Charge Rate ($)", min_value=10.0, max_value=250.0, value=79.90,
                                          step=1.0)
        total_charges = st.number_input("Total Contract Billings ($)", min_value=10.0, max_value=35000.0, value=2450.0,
                                        step=50.0)
        income_level = st.select_slider("Income Bracket Rank", options=["Low", "Medium", "High"], value="High")

    with c3:
        st.markdown(
            "<p style='color:#64748b; font-size:0.75rem; font-weight:700; letter-spacing:0.05em; margin-bottom:12px;'>ENGAGEMENT SPECTRUM</p>",
            unsafe_allow_html=True)
        st.markdown("<p style='color:#f1f5f9; font-size:0.82rem; margin-bottom:2px;'>Account Tenure Months</p>",
                    unsafe_allow_html=True)
        tenure = st.slider("Tenure Months Slider", min_value=1, max_value=72, value=36, label_visibility="collapsed")
        st.markdown(
            "<p style='color:#f1f5f9; font-size:0.82rem; margin-top:14px; margin-bottom:2px;'>Primary Communication Touchpoint</p>",
            unsafe_allow_html=True)
        interaction_type = st.radio("Primary Touchpoint Radio", ["Call Center", "Web Chat", "Email"], horizontal=True,
                                    label_visibility="collapsed")

    with c4:
        st.markdown(
            "<p style='color:#64748b; font-size:0.75rem; font-weight:700; letter-spacing:0.05em; margin-bottom:12px;'>CRM SUPPORT STATUS</p>",
            unsafe_allow_html=True)
        st.markdown("<p style='color:#f1f5f9; font-size:0.82rem; margin-bottom:4px;'>Ticket Lifecycle State</p>",
                    unsafe_allow_html=True)
        resolution_status = st.selectbox("Ticket Status Box", ["Resolved", "Unresolved", "Escalated"],
                                         label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("⚡ Execute Cloud Microservice Analytics Pipeline", type="primary",
                               use_container_width=True)

# --- 4. DECOUPLED NETWORK INFERENCE LOGIC ---
if predict_button:
    # Build clean payload mapped exactly to FastAPI's incoming telemetry validation expectations
    payload = {
        "Age": int(age),
        "Tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "Gender": str(gender),
        "MaritalStatus": str(marital_status),
        "ResolutionStatus": str(resolution_status),
        "InteractionType": str(interaction_type),
        "IncomeLevel": str(income_level)
    }

    with st.spinner("Streaming encrypted telemetry data packet to container nodes..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()

                risk_probability = result["churn_probability"]
                verdict_text = result["risk_verdict"]
                task_bracket = result["action_bracket"]

                # Dynamic Visual Token Interpolation
                if task_bracket == "high":
                    status_color, accent_glow = "#ef4444", "rgba(239, 68, 68, 0.12)"
                elif task_bracket == "medium":
                    status_color, accent_glow = "#f59e0b", "rgba(245, 158, 11, 0.12)"
                else:
                    status_color, accent_glow = "#10b981", "rgba(16, 185, 129, 0.12)"

                # --- 5. INTERACTIVE VIEW RENDERING PANEL ---
                out_left, out_right = st.columns([1.1, 1.4])

                with out_left:
                    st.markdown(f"""
                        <div class="react-card" style="border: 1px solid {status_color}44; background: {accent_glow}; text-align: center; padding: 40px 20px;">
                            <p style="margin:0; color:#64748b; font-size:0.75rem; font-weight:700; letter-spacing:0.1em;">ANALYTICS CORE STATUS VERDICT</p>
                            <h2 style="margin:12px 0 24px 0; color:{status_color}; font-weight:800; font-size:1.75rem; letter-spacing:-0.03em;">{verdict_text}</h2>
                            <div style="width: 80%; height: 1px; background: rgba(255,255,255,0.08); margin: 0 auto 24px auto;"></div>
                            <p style="margin:0; color:#94a3b8; font-size:0.85rem; font-weight:500;">PREDICTIVE ATTRITION PROBABILITY</p>
                            <h1 style="margin:4px 0 0 0; color:#ffffff; font-size:4.2rem; font-weight:800; letter-spacing:-0.04em;">{risk_probability:.1%}</h1>
                        </div>
                    """, unsafe_allow_html=True)

                    # Valuation utility calculations
                    retention_value = 500
                    intervention_cost = 50
                    net_expected_utility = (retention_value * (1 - risk_probability)) - intervention_cost

                    # Interactive Metrics Grid
                    st.markdown('<div class="react-card">', unsafe_allow_html=True)
                    st.markdown(
                        "<p style='color:#64748b; font-size:0.75rem; font-weight:700; letter-spacing:0.05em; margin-bottom:16px;'>FINANCIAL RISK EXPOSURE MATRIX</p>",
                        unsafe_allow_html=True)
                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        st.metric("Intervention Allocation", f"${intervention_cost}.00")
                    with f_col2:
                        st.metric("Net Operational Yield", f"${net_expected_utility:,.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with out_right:
                    # Advanced High-Fidelity Donut Gauge Speedometer Implementation
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_probability * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        number={'suffix': "%", 'font': {'color': '#ffffff', 'family': 'Plus Jakarta Sans', 'size': 24}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                            'bar': {'color': status_color},
                            'bgcolor': "rgba(30, 41, 59, 0.5)",
                            'borderwidth': 0,
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
                        margin=dict(l=30, r=30, t=30, b=10),
                        height=210
                    )

                    st.markdown('<div class="react-card">', unsafe_allow_html=True)
                    st.markdown(
                        "<p style='color:#64748b; font-size:0.75rem; font-weight:700; letter-spacing:0.05em; margin-bottom:4px;'>SYSTEM SPECTRUM TELEMETRY GAUGE</p>",
                        unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                    # CRM Playbook Action Items Panel
                    st.markdown(f'<div class="action-header">⚡ Automated CRM Operational Next-Steps</div>',
                                unsafe_allow_html=True)

                    if task_bracket == "high":
                        st.markdown(
                            f'<div class="task-item high">🚨 <b>Priority 1:</b> Route account ticket directly to specialized high-value retention squad.</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="task-item high">💸 <b>Priority 2:</b> Flag CRM system profile for automatic contract discount pricing credits.</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="task-item high">📞 <b>Priority 3:</b> Trigger direct outbound phone scheduling logs; bypass emails.</div>',
                            unsafe_allow_html=True)
                    elif task_bracket == "medium":
                        st.markdown(
                            f'<div class="task-item medium">⚠️ <b>Priority 1:</b> Inject custom promotional vouchers targeting user preference indexes.</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="task-item medium">🔄 <b>Priority 2:</b> Re-verify support pipeline queues via the active {interaction_type} channel tracker.</div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="task-item low">✅ <b>Routine:</b> Sort client data profiles into monthly engagement email sequences.</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="task-item low">📈 <b>Routine:</b> Account infrastructure stable; enable standardized loyalty cross-sell flags.</div>',
                            unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"Backend Engine Deployment Error (Status Code: {response.status_code})")

        except requests.exceptions.ConnectionError:
            st.error(
                "🚨 Unable to establish a handshake connection with the backend engine container nodes. Verify your Orchestration status via the Makefile system.")