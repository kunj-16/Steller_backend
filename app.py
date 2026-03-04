from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import uvicorn


# ----------------------------
# Load Models
# ----------------------------

reg_model = joblib.load("regression_model.pkl")
clf_model = joblib.load("classification_model.pkl")


# ----------------------------
# FastAPI App
# ----------------------------

app = FastAPI(title="Stellar Prediction API")


# ----------------------------
# CORS Middleware
# ----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Root Endpoint (Health Check)
# ----------------------------

@app.get("/")
def home():
    return {"message": "Stellar Prediction API is running"}


# ----------------------------
# Input Schema
# ----------------------------

class StellarInput(BaseModel):

    koi_period: Optional[float] = None
    koi_duration: Optional[float] = None
    koi_depth: Optional[float] = None
    koi_impact: Optional[float] = None
    koi_model_snr: Optional[float] = None
    koi_num_transits: Optional[float] = None
    koi_ror: Optional[float] = None
    st_teff: Optional[float] = None
    st_logg: Optional[float] = None
    st_met: Optional[float] = None
    st_mass: Optional[float] = None
    st_radius: Optional[float] = None
    st_dens: Optional[float] = None
    teff_err1: Optional[float] = None
    teff_err2: Optional[float] = None
    logg_err1: Optional[float] = None
    logg_err2: Optional[float] = None
    feh_err1: Optional[float] = None
    feh_err2: Optional[float] = None
    mass_err1: Optional[float] = None
    mass_err2: Optional[float] = None
    radius_err1: Optional[float] = None
    radius_err2: Optional[float] = None


# ----------------------------
# Prediction Endpoint
# ----------------------------

@app.post("/predict")
def predict(data: StellarInput):

    input_dict = data.model_dump()

    # Safety Check
    if all(v is None for v in input_dict.values()):
        raise HTTPException(
            status_code=400,
            detail="At least one feature must be provided for prediction."
        )

    try:

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Ensure correct feature order
        input_df = input_df[reg_model.feature_names_in_]

        # -----------------------
        # Regression Prediction
        # -----------------------

        pred_log = reg_model.predict(input_df)
        predicted_radius = float(np.expm1(pred_log)[0])

        # -----------------------
        # Classification
        # -----------------------

        class_pred = int(clf_model.predict(input_df)[0])
        probability = float(clf_model.predict_proba(input_df)[0][1])

        # Label Conversion
        label = "Confirmed" if class_pred == 1 else "False Positive"

        return {
            "predicted_planet_radius": round(predicted_radius, 4),
            "habitability_class": label,
            "habitability_probability": round(probability, 4)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ----------------------------
# Run Server (for Render)
# ----------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000)
