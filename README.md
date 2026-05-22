# RiskHorizon™ Intelligence OS: End-to-End Bank Customer Churn Telemetry System

An enterprise-grade, decoupled machine learning application engineered to predict bank customer attrition risk in real-time. This system features a robust **Scikit-Learn inference pipeline**, packaged inside a production-ready **FastAPI REST architecture**, and exposed through a premium **glassmorphic Streamlit GUI dashboard**. The entire infrastructure is designed for high-availability cloud container execution.

---

## 🏗️ System Architecture Overview

The system is engineered as a decoupled microservices architecture, isolating the mathematical inference engine from the user interface to ensure optimal scalability and independent deployment boundaries.
─────────────────────────────────────────┐
              │       Streamlit GUI Dashboard           │
              │       (Port 8501 / Port 8502)           │
              └────────────────────┬────────────────────┘
                                   │
                    HTTP POST JSON │ (Real-time Telemetry)
                                   ▼
              ┌─────────────────────────────────────────┐
              │       FastAPI Inference Engine          │
              │             (Port 8000)                 │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │      ML Pipeline & Prediction Models    │
              │   (Preprocessor & Random Forest Base)   │
              └─────────────────────────────────────────┘


**Backend (Inference Engine):** A FastAPI REST framework that handles high-throughput predictive scoring, strict input schema parsing with Pydantic, and dynamic classification of customer risk brackets.
* **Frontend (Intelligence OS):** A React-inspired dark fintech interface built on Streamlit with interactive telemetry sliders, financial yield metrics, and custom Plotly gauge analytics.

---

## ⚡ Core Operational Features

* **Pydantic Data Validation:** Enforces strict execution boundaries on incoming account payloads (`Age`, `Tenure`, `Charges`, etc.).
* **Fault-Tolerant Asset Loading:** Features a robust model-loading system with automatic dummy fallbacks to ensure zero-downtime execution environments.
* **Dynamic Risk Brackets:** Automates CRM decision-making by segmenting customer profiles into `CRITICAL RISK PROFILE`, `ELEVATED RISK WARN`, or `STABLE` tiers.
* **Actionable Business Logic:** Calculates direct cost-benefit financial metrics on the fly, yielding estimated net operational values for customer interventions.

---

## 📁 Repository Blueprint

```text
├── api/
│   ├── app.py                # Streamlit Glassmorphic Frontend Dashboard
│   └── main.py               # FastAPI Production Inference Engine
├── data/                     # Analytical Training & Source Data Batches
├── models/                   # Serialized ML Artifacts (*.pkl Storage Files)
├── notebooks/                # Exploratory Data Analysis & Model Training Logs
├── src/                      # Custom Scikit-Learn Preprocessing Transformers
├── docker-compose.yml        # Multi-Container Orchestration Blueprint
├── Dockerfile                # Lightweight Python Base Runtime Container Image
├── README.md                 # System Documentation Blueprint
└── requirements.txt          # Python Framework Core Dependency Register

Local Cloud Environment Installation & Setup
Since this environment operates within an isolated cloud sandbox without default outer Docker layers, run the services natively using the unified project structure:

1. Environmental Standardization
Install all critical data science modules and framework runtimes directly via the dependency index:

Bash
pip install -r requirements.txt
2. Stand Up the Backend Inference Core
Initialize the FastAPI server on port 8000. This command forces it to bind to all available interfaces to bypass proxy barriers:

Bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
Verify the active deployment state by hitting the /health endpoint.

3. Initialize the Fintech Dashboard GUI
Open a secondary terminal workspace tab and launch the Streamlit server on an independent network node mapping:

Bash
python -m streamlit run api/app.py --server.port=8502 --server.address=0.0.0.0
📡 API Telemetry Specification
Core Predictive Inference Endpoint
Endpoint: POST /predict

Content-Type: application/json

Request Payload Specification Example:
JSON
{
  "Age": 42,
  "Tenure": 36,
  "MonthlyCharges": 79.90,
  "TotalCharges": 2450.00,
  "Gender": "Male",
  "MaritalStatus": "Married",
  "ResolutionStatus": "Resolved",
  "InteractionType": "Call Center",
  "IncomeLevel": "High"
}
Response Vector Specification Example:
JSON
{
  "churn_probability": 0.2541,
  "risk_verdict": "ELEVATED RISK WARN",
  "action_bracket": "medium"
}
🛠️ Skills and Tools Demonstrated
This project showcases an integration of software engineering best practices with core machine learning principles:

ML Infrastructure: End-to-end model serializations using joblib, preprocessing pipeline integrations, and vector formatting.

Backend Engineering: RESTful routing designs, HTTP status schema overrides, application lifespan hooks, and validation mechanics.

Frontend Design: Custom CSS component styling injections, state-driven widget data streams, and responsive visual layout structures.

DevOps Foundation: Environment container configuration (Dockerfile), network abstraction mapping, and process isolation setup.


here is screenshot of GUI

              
