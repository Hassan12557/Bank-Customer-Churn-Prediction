"""
Bank Customer Churn Inference API
Framework: FastAPI Production Engine
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- 1. ENVIRONMENT CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Handle potential module paths for custom transformers
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import __main__
try:
    from data_pipeline import DemographicsInteracter
    # Ensure Joblib can safely locate the transformer during unpickling
    setattr(__main__, 'DemographicsInteracter', DemographicsInteracter)
except ImportError:
    pass

# --- 2. INITIALIZE FASTAPI & LOAD ARTIFACTS ---
app = FastAPI(
    title="RiskHorizon™ Inference Engine",
    description="Production-grade REST API for predictive bank customer attrition telemetry.",
    version="1.0.0"
)

# Global variables for model assets
pipeline = None
model = None

@app.on_event("startup")
def load_assets():
    global pipeline, model
    pipeline_path = BASE_DIR / "models" / "preprocessor_pipeline.pkl"
    model_path = BASE_DIR / "models" / "best_churn_model.pkl"
    
    if not pipeline_path.exists() or not model_path.exists():
        raise RuntimeError("🚨 Core ML pipeline artifacts are missing from storage directory.")
        
    pipeline = joblib.load(pipeline_path)
    model = joblib.load(model_path)

# --- 3. PYDANTIC SCHEMAS FOR DATA VALIDATION ---
class TelemetryInput(BaseModel):
    Age: int = Field(..., ge=18, le=100, example=42)
    Tenure: int = Field(..., ge=1, le=72, example=36)
    MonthlyCharges: float = Field(..., ge=10.0, le=250.0, example=79.90)
    TotalCharges: float = Field(..., ge=10.0, le=35000.0, example=2450.0)
    Gender: str = Field(..., example="Male")
    MaritalStatus: str = Field(..., example="Married")
    ResolutionStatus: str = Field(..., example="Resolved")
    InteractionType: str = Field(..., example="Call Center")
    IncomeLevel: str = Field(..., example="High")

class PredictionOutput(BaseModel):
    churn_probability: float = Field(..., example=0.642)
    risk_verdict: str = Field(..., example="CRITICAL RISK PROFILE")
    action_bracket: str = Field(..., example="high")

# --- 4. API ENDPOINTS ---
@st.cache_resource # Mentally mapping system health
@app.get("/health", tags=["System Utility"])
def health_check():
    """Verifies backend operation and core artifact loading states."""
    if pipeline is None or model is None:
        raise HTTPException(status_code=503, detail="Inference engine uninitialized.")
    return {"status": "healthy", "environment": "active_production"}

@app.post("/predict", response_model=PredictionOutput, tags=["Core Inference"])
def predict_churn(payload: TelemetryInput):
    """Processes account telemetry through the Scikit-learn pipeline and calculates churn risk."""
    try:
        # Convert incoming JSON vector smoothly into a Pandas DataFrame shape
        input_data = pd.DataFrame([payload.dict()])
        
        # Transform using preprocessor pipeline
        processed_vector = pipeline.transform(input_data)
        
        # Extract churn risk probability (Target Class 1)
        risk_prob = float(model.predict_proba(processed_vector)[0, 1])
        
        # Classify severity thresholds dynamically
        if risk_prob > 0.50:
            verdict, bracket = "CRITICAL RISK PROFILE", "high"
        elif risk_prob > 0.20:
            verdict, bracket = "ELEVATED RISK WARN", "medium"
        else:
            verdict, bracket = "STABLE / HIGH RETENTION", "low"
            
        return {
            "churn_probability": round(risk_prob, 4),
            "risk_verdict": verdict,
            "action_bracket": bracket
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failure: {str(e)}")
