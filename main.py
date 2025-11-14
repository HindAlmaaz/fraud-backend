from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------- Paths & data loading ----------

BASE_DIR = Path(__file__).resolve().parent

def read_csv(name: str, encoding: str = "utf-8"):
    """Helper to read CSV from the same folder as main.py."""
    path = BASE_DIR / name
    return pd.read_csv(path, encoding=encoding)

# Transactions sometimes cause UTF-8 errors on Render → use latin-1 only for them
transactions_raw = read_csv("Transactions_Clean.csv", encoding="latin-1")
hospitals_df     = read_csv("Hospitals.csv", encoding="utf-8")
drugs_df         = read_csv("Drugs.csv", encoding="utf-8")
doctors_df       = read_csv("Doctors.csv", encoding="utf-8")
patients_df      = read_csv("Patients.csv", encoding="utf-8")

# ---------- Pre-processing & scoring ----------

transactions_raw["date"] = pd.to_datetime(transactions_raw["date"])

# Merge extra info
df = (
    transactions_raw
    .merge(
        drugs_df[["drug_name", "is_controlled", "typical_price"]],
        on="drug_name",
        how="left",
    )
    .merge(
        doctors_df[["doctor_id", "specialty"]],
        on="doctor_id",
        how="left",
    )
    .merge(
        hospitals_df[["hospital_id", "hospital_name"]],
        left_on="pharmacy_id",
        right_on="hospital_id",
        how="left",
    )
)

# Basic financial & risk scores (simple heuristic – just to drive the UI)
df["total_price"] = df["quantity"] * df["unit_price"]

# avoid division by zero
df["financial_score"] = (
    df["total_price"] / df["typical_price"].clip(lower=1)
).fillna(0.0)
df["financial_score"] = (df["financial_score"] / 3).clip(0, 1)

df["dose_score"] = (df["quantity"] / 4).clip(0, 1)
df["controlled_score"] = df["is_controlled"].fillna(0) * 0.7
df["temporal_score"] = 0.2  # constant just so the field exists

df["anomaly_score"] = (
    df["financial_score"] * 0.4
    + df["dose_score"] * 0.3
    + df["controlled_score"] * 0.3
).clip(0, 1)

df["risk_band"] = pd.cut(
    df["anomaly_score"],
    bins=[-0.01, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"],
)

# ---------- Small helpers ----------

def filter_by_hospital_year_quarter(
    hospital_id: str, year: int, quarter: Optional[int]
) -> pd.DataFrame:
    d = df[df["pharmacy_id"] == hospital_id].copy()
    d = d[d["date"].dt.year == year]
    if quarter is not None:
        d = d[d["date"].dt.quarter == quarter]
    return d


def build_fraud_report(
    hospital_id: str, year: int, quarter: Optional[int]
) -> dict:
    d = filter_by_hospital_year_quarter(hospital_id, year, quarter)

    # hospital name (Arabic)
    if not d.empty:
        hospital_name = d["hospital_name"].iloc[0]
    else:
        row = hospitals_df.loc[hospitals_df["hospital_id"] == hospital_id]
        hospital_name = (
            row["hospital_name"].iloc[0] if not row.empty else hospital_id
        )

    total = int(len(d))
    high = int((d["risk_band"] == "High").sum())
    med = int((d["risk_band"] == "Medium").sum())
    low = int((d["risk_band"] == "Low").sum())
    controlled = int(d["is_controlled"].fillna(0).sum())
    active_alerts = high

    overall_band = "LOW"
    if total > 0:
        high_ratio = high / total
        if high_ratio >= 0.4:
            overall_band = "HIGH"
        elif high_ratio >= 0.2:
            overall_band = "MEDIUM"
        else:
            overall_band = "LOW"

    if total > 0:
        risk_distribution = {
            "high": round(high / total * 100, 1),
            "medium": round(med / total * 100, 1),
            "low": round(low / total * 100, 1),
        }
    else:
        risk_distribution = {"high": 0.0, "medium": 0.0, "low": 100.0}

    # simple "last 3 months" trend of high-risk prescriptions
    if not d.empty:
        by_month = (
            d.groupby(d["date"].dt.to_period("M"))
            .apply(lambda g: int((g["risk_band"] == "High").sum()))
            .sort_index()
        )
        counts = list(by_month.values)[-3:]
        while len(counts) < 3:
            counts.insert(0, 0)
    else:
        counts = [0, 0, 0]

    trend = [
        {"month": f"Month {i + 1}", "value": int(v)}
        for i, v in enumerate(counts)
    ]

    summary = (
        f"The current hospital risk level for {hospital_name} "
        f"in {year} is classified as {overall_band}. "
        "This is based on the proportion of prescriptions flagged as high-risk, "
        "controlled substance usage, and pricing anomalies."
    )

    return {
        "hospital_id": hospital_id,
        "hospital_name": hospital_name,
        "year": year,
        "quarter": quarter,
        "total_prescriptions": total,
        "high_risk_cases": high,
        "medium_risk_cases": med,
        "low_risk_cases": low,
        "controlled_drug_use": controlled,
        "active_alerts": active_alerts,
        "risk_band": overall_band,
        "risk_distribution": risk_distribution,
        "trend_last_3_months": trend,
        "summary": summary,
    }


def build_top_cases(
    hospital_id: str, year: int, quarter: Optional[int], limit: int
) -> dict:
    d = filter_by_hospital_year_quarter(hospital_id, year, quarter)
    if d.empty:
        return {
            "hospital_id": hospital_id,
            "year": year,
            "quarter": quarter,
            "cases": [],
        }

    d = d.sort_values("anomaly_score", ascending=False).head(limit)

    cases = []
    for _, row in d.iterrows():
        specialty = (
            str(row["specialty"])
            if "specialty" in row and pd.notna(row["specialty"])
            else None
        )
        cases.append(
            {
                "prescription_id": row["prescription_id"],
                "patient_id": row["patient_id"],
                "doctor_id": row["doctor_id"],
                "doctor_specialty": specialty,
                "drug_name": row["drug_name"],
                "quantity": int(row["quantity"]),
                "total_price": float(row["total_price"]),
                "date": row["date"].strftime("%Y-%m-%d"),
                "anomaly_score": float(round(row["anomaly_score"], 3)),
                "risk_band": str(row["risk_band"]),
                "recommended_action": (
                    "Review prescription manually"
                    if str(row["risk_band"]) == "High"
                    else "Monitor"
                ),
            }
        )

    return {
        "hospital_id": hospital_id,
        "year": year,
        "quarter": quarter,
        "cases": cases,
    }


# ---------- FastAPI app ----------

app = FastAPI(title="Fraud Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production you can restrict to your Netlify URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud backend is running"}


@app.get("/api/hospitals")
def get_hospitals():
    hospitals = [
        {"id": row["hospital_id"], "name": row["hospital_name"]}
        for _, row in hospitals_df.iterrows()
    ]
    return {"hospitals": hospitals}


@app.get("/api/fraud-report")
def fraud_report(
    hospital_id: str,
    year: int,
    quarter: Optional[int] = None,
):
    if hospital_id not in df["pharmacy_id"].unique():
        raise HTTPException(status_code=404, detail="Hospital not found in data")

    report = build_fraud_report(hospital_id, year, quarter)
    return report


@app.get("/api/top-cases")
def top_cases(
    hospital_id: str,
    year: int,
    quarter: Optional[int] = None,
    limit: int = 5,
):
    if hospital_id not in df["pharmacy_id"].unique():
        raise HTTPException(status_code=404, detail="Hospital not found in data")

    payload = build_top_cases(hospital_id, year, quarter, limit)
    return payload
