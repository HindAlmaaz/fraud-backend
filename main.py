# main.py  --- Fraud detection backend (FastAPI)

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------------------------
# 1. Load data with safe encoding (fixes UnicodeDecodeError on Render)
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


def load_csv(name: str) -> pd.DataFrame:
    """Load a CSV from the same folder using a tolerant encoding."""
    return pd.read_csv(
        BASE_DIR / name,
        encoding="latin-1",      # avoids the utf-8 decode error
    )


transactions_df = load_csv("Transactions_Clean.csv")
patients_df = load_csv("Patients.csv")
hospitals_df = load_csv("Hospitals.csv")
doctors_df = load_csv("Doctors.csv")
drugs_df = load_csv("Drugs.csv")

# -------------------------------------------------------------------
# 2. Pre-processing / joins
# -------------------------------------------------------------------

# Ensure datetime and calendar features
transactions_df["date"] = pd.to_datetime(transactions_df["date"])
transactions_df["year"] = transactions_df["date"].dt.year
transactions_df["quarter"] = transactions_df["date"].dt.quarter

# Join doctor specialty
if "specialty" in doctors_df.columns:
    doctors_small = doctors_df[["doctor_id", "specialty"]]
else:
    # if your Doctors.csv uses another col name, change it here
    doctors_small = doctors_df.rename(columns={"doctor_specialty": "specialty"})[
        ["doctor_id", "specialty"]
    ]

transactions_df = transactions_df.merge(
    doctors_small, on="doctor_id", how="left"
)

# Join drug metadata
drugs_small_cols = []
for col in ["drug_name", "is_controlled", "typical_price"]:
    if col in drugs_df.columns:
        drugs_small_cols.append(col)

drugs_small = drugs_df[drugs_small_cols]

transactions_df = transactions_df.merge(
    drugs_small, on="drug_name", how="left"
)

# Clean numeric columns
for col in ["quantity", "unit_price", "total_price", "typical_price"]:
    if col in transactions_df.columns:
        transactions_df[col] = pd.to_numeric(
            transactions_df[col], errors="coerce"
        )

transactions_df["quantity"] = transactions_df["quantity"].fillna(0)

if "typical_price" in transactions_df.columns:
    typical_median = transactions_df["typical_price"].median()
    transactions_df["typical_price"] = transactions_df["typical_price"].fillna(
        typical_median
    )
else:
    transactions_df["typical_price"] = 1.0

transactions_df["unit_price"] = transactions_df["unit_price"].fillna(
    transactions_df["typical_price"]
)
transactions_df["total_price"] = transactions_df["total_price"].fillna(
    transactions_df["unit_price"] * transactions_df["quantity"]
)

if "is_controlled" in transactions_df.columns:
    transactions_df["is_controlled"] = (
        transactions_df["is_controlled"].fillna(False).astype(bool)
    )
else:
    transactions_df["is_controlled"] = False

# Build a quick patient lookup for chronic conditions
if "chronic_conditions" in patients_df.columns:
    chronic_map = patients_df.set_index("patient_id")["chronic_conditions"]
else:
    chronic_map = pd.Series(dtype=object)


# -------------------------------------------------------------------
# 3. Agent scoring logic
# -------------------------------------------------------------------


def compute_agent_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Financial / Temporal / Clinical / Anomaly scores
    for each prescription row in df (copy is returned).
    """

    out = df.copy()

    # -------- Financial score (price outliers) --------
    # ratio of unit_price vs typical_price; 3x or more => close to 1
    ratio = (
        (out["unit_price"] / out["typical_price"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )
    financial = (ratio - 1.0) / 2.0  # 1x -> 0, 3x -> 1
    financial = np.clip(financial, 0.0, 1.0)

    # -------- Temporal score (repeat frequency) --------
    # more refills for same patient+drug => higher score
    if {"patient_id", "drug_name"}.issubset(out.columns):
        counts = out.groupby(["patient_id", "drug_name"])["prescription_id"].transform(
            "count"
        )
        temporal = (counts - 1) / 4.0  # 1 → 0, 5+ → 1
    else:
        temporal = 0.0
    temporal = np.clip(temporal, 0.0, 1.0)

    # -------- Clinical score (controlled drug vs chronic status) --------
    if not chronic_map.empty:
        has_chronic = out["patient_id"].map(chronic_map.notna()).fillna(False)
    else:
        has_chronic = False

    is_ctrl = out["is_controlled"].fillna(False)

    clinical = np.where(is_ctrl & ~has_chronic, 0.8, 0.2)
    clinical = clinical.astype(float)

    # -------- Anomaly score (quantity z-score) --------
    qty = out["quantity"].astype(float)
    if qty.std(ddof=0) > 0:
        z = (qty - qty.mean()) / qty.std(ddof=0)
        anomaly = (z - 1.0) / 2.0  # 1σ→0, 3σ→1
    else:
        anomaly = 0.0
    anomaly = np.clip(anomaly, 0.0, 1.0)

    # -------- Final fused score (meta-learner-lite) --------
    final = (
        0.35 * financial + 0.25 * temporal + 0.25 * clinical + 0.15 * anomaly
    )
    final = np.clip(final, 0.0, 1.0)

    # Risk band
    bands = np.select(
        [final >= 0.75, final >= 0.40],
        ["High", "Medium"],
        default="Low",
    )

    actions = np.where(
        bands == "High",
        "Needs investigation",
        np.where(bands == "Medium", "Review required", "Low risk"),
    )

    out["financial_score"] = np.round(financial, 3)
    out["temporal_score"] = np.round(temporal, 3)
    out["clinical_score"] = np.round(clinical, 3)
    out["anomaly_score"] = np.round(anomaly, 3)
    out["final_fraud_score"] = np.round(final, 3)
    out["risk_band"] = bands
    out["recommended_action"] = actions

    return out


# -------------------------------------------------------------------
# 4. FastAPI app + CORS
# -------------------------------------------------------------------

app = FastAPI(title="Hospital Pharmacy Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to your Netlify domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud backend is running"}


# -------------------------------------------------------------------
# 5. Helper to filter by hospital/year/quarter
# -------------------------------------------------------------------


def filter_slice(hospital_id: str, year: int, quarter: Optional[int]) -> pd.DataFrame:
    sub = transactions_df[transactions_df["hospital_id"] == hospital_id].copy()

    if year is not None:
        sub = sub[sub["year"] == year]

    if quarter is not None:
        sub = sub[sub["quarter"] == quarter]

    return sub


# -------------------------------------------------------------------
# 6. /api/fraud-report  (summary KPIs)
# -------------------------------------------------------------------


@app.get("/api/fraud-report")
def fraud_report(
    hospital_id: str = Query(..., description="Hospital ID, e.g., H001"),
    year: int = Query(..., description="Year, e.g., 2025"),
    quarter: Optional[int] = Query(
        None, ge=1, le=4, description="Quarter (1–4). If omitted, use all quarters."
    ),
):
    sub = filter_slice(hospital_id, year, quarter)

    if sub.empty:
        raise HTTPException(status_code=404, detail="No data for this filter")

    scored = compute_agent_scores(sub)

    total_prescriptions = len(scored)
    high_risk_cases = int((scored["risk_band"] == "High").sum())
    medium_risk_cases = int((scored["risk_band"] == "Medium").sum())
    low_risk_cases = int((scored["risk_band"] == "Low").sum())
    controlled_drug_use = int(scored["is_controlled"].sum())

    # Simple definition: alerts when Medium or High
    active_alerts = int(
        (scored["risk_band"].isin(["High", "Medium"])).sum()
    )

    # Overall risk level from proportion of High+Medium
    alert_ratio = (high_risk_cases + medium_risk_cases) / max(
        total_prescriptions, 1
    )
    if alert_ratio >= 0.35:
        risk_level = "HIGH"
    elif alert_ratio >= 0.15:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Trend over 3 time buckets inside the slice (for the line chart)
    scored = scored.sort_values("date")
    scored["bucket"] = pd.qcut(
        scored.index, q=3, labels=["Month 1", "Month 2", "Month 3"]
    )

    trend = (
        scored.groupby("bucket")["risk_band"]
        .apply(lambda s: (s.isin(["High", "Medium"])).sum())
        .reset_index()
        .rename(columns={"risk_band": "fraud_cases"})
    )
    trend_records = [
        {"month": row["bucket"], "value": int(row["fraud_cases"])}
        for _, row in trend.iterrows()
    ]

    # Top suspicious drugs (for bar chart)
    top_drugs = (
        scored.groupby("drug_name")["final_fraud_score"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
        .reset_index()
    )
    top_drugs_records = [
        {"drug_name": row["drug_name"], "avg_fraud_score": round(row["final_fraud_score"], 3)}
        for _, row in top_drugs.iterrows()
    ]

    return {
        "hospital_id": hospital_id,
        "year": year,
        "quarter": quarter,
        "total_prescriptions": total_prescriptions,
        "high_risk_cases": high_risk_cases,
        "medium_risk_cases": medium_risk_cases,
        "low_risk_cases": low_risk_cases,
        "controlled_drug_use": controlled_drug_use,
        "active_alerts": active_alerts,
        "risk_level": risk_level,
        "trend_last_3_buckets": trend_records,
        "top_suspicious_drugs": top_drugs_records,
    }


# -------------------------------------------------------------------
# 7. /api/top-cases  (detailed suspicious prescriptions)
# -------------------------------------------------------------------


@app.get("/api/top-cases")
def top_cases(
    hospital_id: str = Query(...),
    year: int = Query(...),
    quarter: Optional[int] = Query(None, ge=1, le=4),
    limit: int = Query(10, ge=1, le=50),
):
    sub = filter_slice(hospital_id, year, quarter)

    if sub.empty:
        raise HTTPException(status_code=404, detail="No data for this filter")

    scored = compute_agent_scores(sub)

    # Sort by final fraud score descending
    scored = scored.sort_values("final_fraud_score", ascending=False).head(limit)

    records = []
    for _, row in scored.iterrows():
        records.append(
            {
                "prescription_id": row.get("prescription_id"),
                "patient_id": row.get("patient_id"),
                "doctor_id": row.get("doctor_id"),
                "doctor_specialty": row.get("specialty"),
                "drug_name": row.get("drug_name"),
                "quantity": float(row.get("quantity", 0)),
                "date": row.get("date").strftime("%Y-%m-%d")
                if pd.notnull(row.get("date"))
                else None,
                "financial_score": float(row.get("financial_score", 0.0)),
                "temporal_score": float(row.get("temporal_score", 0.0)),
                "clinical_score": float(row.get("clinical_score", 0.0)),
                "anomaly_score": float(row.get("anomaly_score", 0.0)),
                "final_fraud_score": float(row.get("final_fraud_score", 0.0)),
                "risk_band": row.get("risk_band"),
                "recommended_action": row.get("recommended_action"),
            }
        )

    return {
        "hospital_id": hospital_id,
        "year": year,
        "quarter": quarter,
        "cases": records,
    }

# -------------------------------------------------------------------
# Note: DO NOT call uvicorn.run() here.
# Render will start it with: uvicorn main:app --host 0.0.0.0 --port $PORT
# -------------------------------------------------------------------
