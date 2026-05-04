import logging
import os
import pickle
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")

def _load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# ── Model artefacts (loaded once at startup) ──────────────────────────────────
model: object = None
scaler: object = None
feature_columns: list = None
threshold: float = 0.31


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, feature_columns, threshold

    for model_file in ("calibrated_model.pkl", "advanced_ensemble.pkl", "ensemble_model.pkl"):
        model_path = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(model_path):
            model = _load(model_path)
            logger.info("Loaded model: %s", model_file)
            break

    for scaler_file in (os.path.join(MODELS_DIR, "scaler.pkl"),
                        os.path.join(DATA_DIR,   "scaler.pkl")):
        if os.path.exists(scaler_file):
            scaler = _load(scaler_file)
            logger.info("Loaded scaler from %s", scaler_file)
            break

    for feat_file in (os.path.join(MODELS_DIR, "feature_columns.pkl"),
                      os.path.join(DATA_DIR,   "feature_names_adv.pkl"),
                      os.path.join(DATA_DIR,   "feature_names.pkl")):
        if os.path.exists(feat_file):
            feature_columns = _load(feat_file)
            logger.info("Loaded %d feature columns from %s", len(feature_columns), feat_file)
            break

    thresh_path = os.path.join(MODELS_DIR, "threshold.pkl")
    if os.path.exists(thresh_path):
        threshold = float(_load(thresh_path))
        logger.info("Loaded threshold: %.3f", threshold)

    if model is None:
        logger.error("No model file found in %s", MODELS_DIR)
    if scaler is None:
        logger.warning("No scaler found — predictions will use unscaled features")
    if feature_columns is None:
        logger.warning("No feature columns file found — column alignment disabled")

    yield


app = FastAPI(
    title="Churn Prediction & Causal ROI API",
    description="Predicts customer churn probability using a calibrated ensemble (LR + GBM + XGBoost) with causal-inference-informed segmentation.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Input schema (raw Telco fields — no pre-engineering needed) ───────────────
class CustomerData(BaseModel):
    tenure: int                   = Field(..., ge=0, le=72,    description="Months with company")
    MonthlyCharges: float         = Field(..., ge=0,            description="Monthly bill (USD)")
    TotalCharges: float           = Field(..., ge=0,            description="Lifetime spend (USD)")
    gender: Literal["Male", "Female"] = "Male"
    SeniorCitizen: Literal[0, 1]  = 0
    Partner: Literal["Yes", "No"] = "No"
    Dependents: Literal["Yes", "No"] = "No"
    PhoneService: Literal["Yes", "No"] = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "DSL"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    OnlineBackup: Literal["Yes", "No", "No internet service"] = "No"
    DeviceProtection: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "No"
    StreamingMovies: Literal["Yes", "No", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = "Electronic check"

    @field_validator("TotalCharges")
    @classmethod
    def total_not_less_than_monthly(cls, v, info):
        monthly = info.data.get("MonthlyCharges", 0)
        tenure  = info.data.get("tenure", 0)
        if tenure > 0 and v < monthly:
            raise ValueError("TotalCharges cannot be less than MonthlyCharges for a customer with tenure > 0")
        return v


# ── Feature engineering (mirrors notebook pipeline) ───────────────────────────
def _engineer_features(customer: CustomerData) -> pd.DataFrame:
    row = customer.model_dump()
    df  = pd.DataFrame([row])

    # Numeric derived features
    df["AvgMonthlySpend"] = (
        df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    ).fillna(0)

    services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
    df["HasMultipleServices"] = df[services].apply(lambda x: (x == "Yes").sum(), axis=1)

    contract_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
    df["ContractMonths"] = df["Contract"].map(contract_map)
    df["Contract_Tenure_Interaction"] = df["ContractMonths"] * df["tenure"]

    # Advanced interaction features (SHAP-confirmed top drivers)
    is_mtm    = (df["Contract"] == "Month-to-month").astype(int)
    is_fiber  = (df["InternetService"] == "Fiber optic").astype(int)
    is_early  = (df["tenure"] <= 6).astype(int)

    df["price_stress"]       = df["MonthlyCharges"] * is_mtm
    df["price_tenure_risk"]  = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["fiber_contract_risk"]= is_fiber * is_mtm
    df["early_customer"]     = is_early
    df["fiber_new_customer"] = is_fiber * is_early

    # Tenure group
    def _tenure_group(t):
        if t <= 12: return "0-12"
        if t <= 24: return "13-24"
        if t <= 48: return "25-48"
        if t <= 60: return "49-60"
        return "60+"
    df["TenureGroup"] = df["tenure"].apply(_tenure_group)

    # One-hot encode — drop_first=True matches training
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align to training feature set
    if feature_columns is not None:
        df = df.reindex(columns=feature_columns, fill_value=0)

    return df


# ── Prediction helper ──────────────────────────────────────────────────────────
def _predict(df: pd.DataFrame) -> tuple[float, bool]:
    X = df.values
    if scaler is not None:
        X = scaler.transform(X)
    prob  = float(model.predict_proba(X)[0][1])
    label = prob >= threshold
    return prob, label


def _segment(prob: float, uplift: float | None = None) -> str:
    high_risk = prob >= threshold
    # Without a live uplift score, fall back to probability-based heuristic
    if high_risk:
        return "Persuadable"   # primary intervention target
    return "Sure Thing" if prob < 0.2 else "Sleeping Dog"


def _intervention(prob: float, contract: str, internet: str) -> str:
    if prob < threshold:
        return "No action required"
    if contract == "Month-to-month":
        return "Offer annual contract discount"
    if internet == "Fiber optic":
        return "Proactive support outreach + service-quality review"
    return "Loyalty retention package"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "n_features": len(feature_columns) if feature_columns else None,
        "threshold": threshold,
    }


@app.get("/", tags=["ops"])
def root():
    return {"message": "Churn Prediction API — see /docs for usage, /health for status."}


@app.post("/predict", tags=["prediction"])
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    try:
        df   = _engineer_features(customer)
        prob, is_churn = _predict(df)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    risk_level = "High" if prob >= 0.6 else ("Medium" if prob >= threshold else "Low")
    segment    = _segment(prob)
    action     = _intervention(prob, customer.Contract, customer.InternetService)

    logger.info(
        "prediction | tenure=%d contract=%s internet=%s prob=%.3f risk=%s segment=%s",
        customer.tenure, customer.Contract, customer.InternetService,
        prob, risk_level, segment,
    )

    return {
        "churn_probability": round(prob, 4),
        "churn_predicted":   bool(is_churn),
        "risk_level":        risk_level,
        "segment":           segment,
        "recommended_intervention": action,
        "model_threshold":   threshold,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
