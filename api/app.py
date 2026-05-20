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

# Premium React-Style CSS Injection Framework
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;600;800&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
            background-color: #0b0f19 !important;
        }

        .react-card {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        .react-card:hover {
            border: 1px solid rgba(59, 130, 246, 0.3);
            box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.1);
        }

        .react-hero-title {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .react-hero-sub {
            color: #64748b;
            font-size: 1.1rem;
            font-weight: 400;
            margin-top: 5px;
            margin-bottom: 30px;
        }

        .action-header {
            font-weight: 600;
            color: #f1f5f9;
            font-size: 1.1rem;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .task-item {
            background: rgba(15, 23, 42, 0.6);
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 4px solid #64748b;
            margin-bottom: 8px;
            font-size: 0.9rem;
            color: #cbd5e1;
        }
        .task-item.high { border-left-color: #f87171; background: rgba(248, 113, 113, 0.05); }
        .task-item.medium { border-left-color: #fbbf24; background: rgba(251, 191, 36, 0.05); }
        .task-item.low { border-left-color: #34d399; background: rgba(52, 211, 153, 0.05); }
    </style>
""", unsafe_allow_html=True)

# --- 2. FINTECH HERO DISPLAY SECTION ---
st.markdown('<h1 class="react-hero-title">RiskHorizon™ Intelligence OS</h1>', unsafe_allow_html=True)
st.markdown('<p class="react-hero-sub">Next-Gen Predictive Customer Attrition Engine • Active Environment v3.1.4</p>',
            unsafe_allow_html=True)

# --- 3. HIGH-FIDELITY LAYOUT FORM COMPONENT ---
with st.container():
    st.markdown('<div class="react-card">', unsafe_allow_html=True)
    st.markdown(
        "<h3 style='margin-top:0; color:#f1f5f9; font-weight:600; font-size:1.3rem;'>📊 Live Account Telemetry Input</h3>",
        unsafe_allow_html=True)

    # Grid layout mapping
    c1, c2, c3, c4 = st.columns([1, 1.2, 1.2, 1])

    with c1:
        st.markdown(
            "<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:5px; font-weight:600;'>ACCOUNT HOLDER</p>",
            unsafe_allow_html=True)
        age = st.number_input("Age Profile", min_value=18, max_value=95, value=42)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Layout", ["Single", "Married", "Divorced"])

    with c2:
        st.markdown(
            "<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:5px; font-weight:600;'>FINANCIAL METRICS</p>",
            unsafe_allow_html=True)
        monthly_charges = st.number_input("Monthly Charge ($)", min_value=10.0, max_value=250.0, value=79.90, step=1.0)
        total_charges = st.number_input("Total Billings ($)", min_value=10.0, max_value=35000.0, value=2450.0,
                                        step=50.0)
        income_level = st.select_slider("Income Bracket", options=["Low", "Medium", "High"], value="High")

    with c3:
        st.markdown(
            "<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:5px; font-weight:600;'>ENGAGEMENT METRICS</p>",
            unsafe_allow_html=True)
        st.markdown("<p style='color:#f1f5f9; font-size:0.85rem; margin-top:10px;'>Tenure Months</p>",
                    unsafe_allow_html=True)
        tenure = st.slider("Tenure Months Slider", min_value=1, max_value=72, value=36, label_visibility="collapsed")
        st.markdown("<p style='color:#f1f5f9; font-size:0.85rem; margin-top:15px;'>Primary Touchpoint</p>",
                    unsafe_allow_html=True)
        interaction_type = st.radio("Primary Touchpoint Radio", ["Call Center", "Web Chat", "Email"], horizontal=True,
                                    label_visibility="collapsed")

    with c4:
        st.markdown(
            "<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:5px; font-weight:600;'>SUPPORT MATRIX</p>",
            unsafe_allow_html=True)
        st.markdown("<p style='color:#f1f5f9; font-size:0.85rem;'>Ticket Status</p>", unsafe_allow_html=True)
        resolution_status = st.selectbox("Ticket Status Box", ["Resolved", "Unresolved", "Escalated"],
                                         label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("⚡ Run Real-Time Cloud API Prediction", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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

    with st.spinner("Communicating with RiskHorizon™ Cloud Engine..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()

                risk_probability = result["churn_probability"]
                verdict_text = result["risk_verdict"]
                task_bracket = result["action_bracket"]

                # Assign status colors based on the microservice's processed verdict tier
                if task_bracket == "high":
                    status_color, accent_glow = "#ef4444", "rgba(239, 68, 68, 0.15)"
                elif task_bracket == "medium":
                    status_color, accent_glow = "#fbbf24", "rgba(251, 191, 36, 0.15)"
                else:
                    status_color, accent_glow = "#10b981", "rgba(16, 185, 129, 0.15)"

                # --- 5. INTERACTIVE VIEW RENDERING PANEL ---
                out_left, out_right = st.columns([1.1, 1.4])

                with out_left:
                    st.markdown(f"""
                        <div class="react-card" style="border: 1px solid {status_color}33; background: {accent_glow}; text-align: center;">
                            <p style="margin:0; color:#94a3b8; font-size:0.85rem; font-weight:600; letter-spacing:0.05em;">SYSTEM VERDICT STATUS</p>
                            <h2 style="margin:5px 0 15px 0; color:{status_color}; font-weight:800; font-size:1.6rem; letter-spacing:-0.02em;">{verdict_text}</h2>
                            <hr style="border:0; border-top:1px solid rgba(255,255,255,0.08); margin:15px 0;">
                            <p style="margin:0; color:#64748b; font-size:0.8rem;">MODEL CONFIDENCE SCORE</p>
                            <h1 style="margin:0; color:#ffffff; font-size:3.5rem; font-weight:800;">{risk_probability:.1%}</h1>
                        </div>
                    """, unsafe_allow_html=True)

                    # Valuation utility calculations
                    retention_value = 500
                    intervention_cost = 50
                    net_expected_utility = (retention_value * (1 - risk_probability)) - intervention_cost

                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        st.metric("Intervention Cost", f"${intervention_cost}.00")
                    with f_col2:
                        st.metric("Net Predicted Yield", f"${net_expected_utility:,.2f}")

                with out_right:
                    # Render Horizonal Spectrum Chart Element
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[risk_probability * 100], y=['Risk'],
                        orientation='h', width=0.4,
                        marker=dict(color=status_color, line=dict(color='rgba(0,0,0,0)', width=0)),
                        hoverinfo='none'
                    ))
                    fig.update_layout(
                        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, color='#475569'),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15, 23, 42, 0.6)',
                        margin=dict(l=0, r=0, t=10, b=10), height=70, showlegend=False
                    )

                    st.markdown('<div class="react-card">', unsafe_allow_html=True)
                    st.markdown(
                        "<p style='color:#94a3b8; font-size:0.85rem; font-weight:600; margin:0 0 5px 0;'>RISK SPECTRUM POSITIONING</p>",
                        unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                    # CRM Playbook Action Items Panel
                    st.markdown('<div class="action-header">⚡ CRM Actionable Next-Best-Steps Panel</div>',
                                unsafe_allow_html=True)

                    if task_bracket == "high":
                        st.markdown(
                            f'<div class="task-item high">🚨 <b>Priority 1:</b> Route immediately to senior accounts resolution squad.</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="task-item high">💸 <b>Priority 2:</b> Flag account for a high-value retention fee reduction credit.</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="task-item high">📞 <b>Priority 3:</b> Bypass automated emails; schedule a high-touch outbound account review.</div>',
                            unsafe_allow_html=True)
                    elif task_bracket == "medium":
                        st.markdown(
                            f'<div class="task-item medium">⚠️ <b>Priority 1:</b> Deploy customized email retention incentives based on user channel usage.</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="task-item medium">🔄 <b>Priority 2:</b> Re-verify open ticket logs via the {interaction_type} pipeline queue.</div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="task-item low">✅ <b>Routine:</b> Group account into standard monthly loyalty engagement workflows.</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="task-item low">📈 <b>Routine:</b> Account stable; cross-sell additional system features if applicable.</div>',
                            unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"Backend Engine Error (Status Code: {response.status_code})")

        except requests.exceptions.ConnectionError:
            st.error(
                "🚨 Unable to connect to the Inference Engine API. Check if the backend Docker container is running.")