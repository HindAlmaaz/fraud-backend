from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------- CONFIG & DATA LOADING ----------

BASE_DIR = Path(__file__).resolve().parent


def load_csv(name: str) -> pd.DataFrame:
    """
    Load a CSV from the repo root, using a tolerant encoding.
    This is Render-friendly and avoids utf-8 decode errors.
    """
    path = BASE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, encoding="latin-1")


# Load base tables
transactions_df = load_csv("Transactions_Clean.csv")
patients_df = load_csv("Patients.csv")
hospitals_df = load_csv("Hospitals.csv")
drugs_df = load_csv("Drugs.csv")
doctors_df = load_csv("Doctors.csv")

# ---------- PREPROCESSING ----------

df = transactions_df.copy()

# Dates / year / quarter
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["quarter"] = df["date"].dt.quarter

# Make sure score columns exist
score_cols = ["financial_score", "temporal_score", "clinical_score", "anomaly_score"]
for col in score_cols:
    if col not in df.columns:
        df[col] = 0.0

df[score_cols] = df[score_cols].fillna(0.0)

# Final fraud score
if "final_fraud_score" not in df.columns:
    df["final_fraud_score"] = df[score_cols].mean(axis=1)
df["final_fraud_score"] = df["final_fraud_score"].fillna(0.0)


def risk_band_from_score(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.3:
        return "Medium"
    return "Low"


if "risk_band" not in df.columns:
    df["risk_band"] = df["final_fraud_score"].apply(risk_band_from_score)
else:
    df["risk_band"] = df["risk_band"].fillna(
        df["final_fraud_score"].apply(risk_band_from_score)
    )

# ---------- FASTAPI APP ----------

app = FastAPI(title="Fraud Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for prototype; lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- HELPERS ----------


def filter_df(hospital_id: str, year: int, quarter: Optional[int]) -> pd.DataFrame:
    sub = df[df["hospital_id"] == hospital_id]
    sub = sub[sub["year"] == year]
    if quarter is not None:
        sub = sub[sub["quarter"] == quarter]
    return sub


def safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


# ---------- ROUTES ----------


@app.get("/")
def health():
    return {"status": "ok", "message": "Fraud backend is running"}


@app.get("/api/hospitals")
def list_hospitals():
    """
    Simple helper for the dropdown.
    """
    if "hospital_id" not in hospitals_df.columns:
        raise HTTPException(500, "Hospitals.csv missing hospital_id column")

    cols = [c for c in hospitals_df.columns if c in ("hospital_id", "hospital_name")]
    records = hospitals_df[cols].to_dict(orient="records")
    return {"hospitals": records}


@app.get("/api/fraud-report")
def fraud_report(
    hospital_id: str,
    year: int,
    quarter: Optional[int] = None,
):
    """
    Summary metrics used by the dashboard cards and charts.
    """
    sub = filter_df(hospital_id, year, quarter)

    if sub.empty:
        # Return a "no data" structure that the frontend can still render
        return {
            "hospital_id": hospital_id,
            "year": year,
            "quarter": quarter,
            "total_prescriptions": 0,
            "high_risk_cases": 0,
            "medium_risk_cases": 0,
            "low_risk_cases": 0,
            "controlled_drug_use": 0,
            "active_alerts": 0,
            "fraud_trend_3_months": [
                {"month": "Month 1", "value": 0},
                {"month": "Month 2", "value": 0},
                {"month": "Month 3", "value": 0},
            ],
            "top_suspicious_drugs": [],
            "risk_distribution": [
                {"label": "High", "value": 0},
                {"label": "Medium", "value": 0},
                {"label": "Low", "value": 0},
            ],
            "summary": "No prescriptions available for this filter.",
        }

    # Basic counts
    total_prescriptions = int(len(sub))
    high_risk = int((sub["risk_band"] == "High").sum())
    medium_risk = int((sub["risk_band"] == "Medium").sum())
    low_risk = int((sub["risk_band"] == "Low").sum())

    # Controlled drug use (where is_controlled is True in Drugs.csv)
    controlled = 0
    if "is_controlled" in drugs_df.columns:
        controlled_drug_names = drugs_df[
            drugs_df["is_controlled"] == True  # noqa: E712
        ]["drug_name"].unique()
        controlled = int(sub["drug_name"].isin(controlled_drug_names).sum())

    # Very simple "active alerts" proxy: high-risk prescriptions
    active_alerts = high_risk

    # Fraud trend over last 3 months in this selection
    sub_sorted = sub.sort_values("date")
    sub_sorted["month_label"] = sub_sorted["date"].dt.to_period("M").astype(str)
    trend_series = (
        sub_sorted.groupby("month_label")["final_fraud_score"].mean().tail(3)
    )

    trend_points: List[Dict[str, Any]] = []
    for i, (month_label, val) in enumerate(trend_series.items(), start=1):
        trend_points.append({"month": f"Month {i}", "value": round(safe_float(val), 2)})

    # Pad to exactly 3 points
    while len(trend_points) < 3:
        trend_points.insert(0, {"month": f"Month {3 - len(trend_points)}", "value": 0})

    # Top suspicious drugs
    top_drugs_df = (
        sub.groupby("drug_name")["final_fraud_score"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
        .reset_index()
    )
    top_drugs = [
        {
            "drug_name": row["drug_name"],
            "avg_fraud_score": round(safe_float(row["final_fraud_score"]), 3),
        }
        for _, row in top_drugs_df.iterrows()
    ]

    # Risk distribution (percentage of prescriptions)
    risk_counts = sub["risk_band"].value_counts()
    total_risk = int(risk_counts.sum())
    if total_risk == 0:
        risk_distribution = [
            {"label": "High", "value": 0},
            {"label": "Medium", "value": 0},
            {"label": "Low", "value": 0},
        ]
    else:
        def pct(label: str) -> float:
            return round(
                safe_float(risk_counts.get(label, 0)) / total_risk * 100, 1
            )

        risk_distribution = [
            {"label": "High", "value": pct("High")},
            {"label": "Medium", "value": pct("Medium")},
            {"label": "Low", "value": pct("Low")},
        ]

    # Simple summary text for the bottom paragraph
    risk_level = "LOW"
    if high_risk > 0:
        risk_level = "HIGH"
    elif medium_risk > 0:
        risk_level = "MEDIUM"

    summary = (
        f"The current hospital risk level is classified as {risk_level} "
        f"based on {high_risk} high-risk, {medium_risk} medium-risk and "
        f"{low_risk} low-risk prescriptions in {total_prescriptions} total prescriptions."
    )

    return {
        "hospital_id": hospital_id,
        "year": year,
        "quarter": quarter,
        "total_prescriptions": total_prescriptions,
        "high_risk_cases": high_risk,
        "medium_risk_cases": medium_risk,
        "low_risk_cases": low_risk,
        "controlled_drug_use": controlled,
        "active_alerts": active_alerts,
        "fraud_trend_3_months": trend_points,
        "top_suspicious_drugs": top_drugs,
        "risk_distribution": risk_distribution,
        "summary": summary,
    }


@app.get("/api/top-cases")
def top_cases(
    hospital_id: str,
    year: int,
    quarter: Optional[int] = None,
    limit: int = 5,
):
    """
    Detailed list of top suspicious prescriptions + small risk distribution.
    Used by your 'Top 10 suspicious cases' design.
    """
    sub = filter_df(hospital_id, year, quarter)

    if sub.empty:
        return {
            "hospital_id": hospital_id,
            "year": year,
            "quarter": quarter,
            "cases": [],
            "risk_distribution": [],
            "message": "No data available for this selection.",
        }

    scored = sub.copy()
    scored = scored.sort_values("final_fraud_score", ascending=False).head(limit)

    # Use existing risk_band as bucket
    scored["bucket"] = scored["risk_band"]

    # SAFE groupby: only if not empty
    if scored.empty:
        risk_distribution = []
    else:
        risk_counts = (
            scored.groupby("bucket")["risk_band"]
            .count()
            .reset_index(name="count")
        )
        total = int(risk_counts["count"].sum())

        risk_distribution = []
        for _, row in risk_counts.iterrows():
            c = int(row["count"])
            percent = round(c / total * 100, 1) if total > 0 else 0.0
            risk_distribution.append(
                {
                    "bucket": row["bucket"],
                    "count": c,
                    "percent": percent,
                }
            )

    def recommend_action(band: str) -> str:
        if band == "High":
            return "Needs investigation"
        if band == "Medium":
            return "Review required"
        return "Low risk"

    cases = []
    for _, row in scored.iterrows():
        cases.append(
            {
                "prescription_id": row.get("prescription_id"),
                "patient_id": row.get("patient_id"),
                "doctor_id": row.get("doctor_id"),
                "drug_name": row.get("drug_name"),
                "quantity": int(row.get("quantity", 0)),
                "date": row["date"].strftime("%Y-%m-%d"),
                "financial_score": safe_float(row.get("financial_score")),
                "temporal_score": safe_float(row.get("temporal_score")),
                "clinical_score": safe_float(row.get("clinical_score")),
                "anomaly_score": safe_float(row.get("anomaly_score")),
                "final_fraud_score": round(
                    safe_float(row.get("final_fraud_score")), 3
                ),
                "risk_band": row.get("risk_band"),
                "recommended_action": recommend_action(row.get("risk_band")),
            }
        )

    return {
        "hospital_id": hospital_id,
        "year": year,
        "quarter": quarter,
        "cases": cases,
        "risk_distribution": risk_distribution,
    }
