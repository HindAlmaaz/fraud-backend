# main.py
from pathlib import Path
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# ---------------------------------------------------------------------
# Paths & CSV loading
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


def load_csv(name: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Generic CSV loader, default UTF-8."""
    return pd.read_csv(BASE_DIR / name, encoding=encoding)


# ⚠️ Only Transactions_Clean needs latin-1 (this avoided your utf-8 error on Render)
transactions_df = load_csv("Transactions_Clean.csv", encoding="latin-1")
patients_df = load_csv("Patients.csv")
hospitals_df = load_csv("Hospitals.csv")
drugs_df = load_csv("Drugs.csv")
doctors_df = load_csv("Doctors.csv")

# ---------------------------------------------------------------------
# Basic cleaning / types
# ---------------------------------------------------------------------
# Ensure column names that we rely on exist
REQUIRED_TX_COLS = [
    "prescription_id",
    "patient_id",
    "doctor_id",
    "hospital_id",
    "drug_name",
    "quantity",
    "unit_price",
    "total_price",
    "date",
]
missing = [c for c in REQUIRED_TX_COLS if c not in transactions_df.columns]
if missing:
    raise RuntimeError(f"Missing columns in Transactions_Clean.csv: {missing}")

# Parse dates
transactions_df["date"] = pd.to_datetime(transactions_df["date"], errors="coerce")

# Add year / quarter helpers
transactions_df["year"] = transactions_df["date"].dt.year
transactions_df["quarter"] = transactions_df["date"].dt.quarter

# Ensure useful columns exist
if "is_fraudulent" not in transactions_df.columns:
    transactions_df["is_fraudulent"] = 0

if "is_controlled" not in drugs_df.columns:
    # if your CSV has 0/1 or True/False, pandas will keep them; if not, default False
    drugs_df["is_controlled"] = False

# ---------------------------------------------------------------------
# Merge + scoring
# ---------------------------------------------------------------------
merged = (
    transactions_df
    .merge(patients_df, on="patient_id", how="left", suffixes=("", "_patient"))
    .merge(doctors_df, on="doctor_id", how="left", suffixes=("", "_doctor"))
    .merge(hospitals_df, on="hospital_id", how="left", suffixes=("", "_hospital"))
    .merge(drugs_df, on="drug_name", how="left", suffixes=("", "_drug"))
)

# Simple financial score: how far from typical price
def compute_financial_score(row):
    try:
        typical = float(row.get("typical_price", 0) or 0)
        qty = float(row.get("quantity", 0) or 0)
        total = float(row.get("total_price", 0) or 0)
    except Exception:
        return 0.0

    expected = typical * qty
    if expected <= 0:
        return 0.0

    ratio = total / expected
    if ratio <= 1:
        return 0.0
    # cap at 1
    return min(1.0, (ratio - 1.0))


# Temporal score: many prescriptions of same drug for same patient in short time
def compute_temporal_score(df: pd.DataFrame) -> pd.Series:
    df_sorted = df.sort_values(["patient_id", "drug_name", "date"])
    df_sorted["prev_date"] = df_sorted.groupby(
        ["patient_id", "drug_name"]
    )["date"].shift(1)
    days = (df_sorted["date"] - df_sorted["prev_date"]).dt.days
    score = days.fillna(9999).apply(
        lambda d: 1.0 if d <= 7 else (0.5 if d <= 30 else 0.0)
    )
    df_sorted["temporal_score"] = score
    return df_sorted.sort_index()["temporal_score"]


# Clinical score: controlled + chronic disease mismatch ⇒ higher risk
def compute_clinical_score(row):
    is_ctrl = bool(row.get("is_controlled", False))
    chronic = str(row.get("chronic_conditions", "") or "").lower()
    drug = str(row.get("drug_name", "") or "").lower()

    base = 0.0
    if is_ctrl:
        base += 0.4

    # tiny heuristic examples – adjust to your real logic if you like
    pain_keywords = ["morphine", "fentanyl", "oxycodone"]
    if any(k in drug for k in pain_keywords) and "cancer" not in chronic:
        base += 0.4

    diabetes_keywords = ["insulin", "metformin"]
    if any(k in drug for k in diabetes_keywords) and "diabetes" not in chronic:
        base += 0.4

    return min(1.0, base)


# Anomaly score: combination of the above + mark known fraud
merged["financial_score"] = merged.apply(compute_financial_score, axis=1)
merged["temporal_score"] = compute_temporal_score(merged)
merged["clinical_score"] = merged.apply(compute_clinical_score, axis=1)

# if label exists, bump the score
merged["anomaly_score"] = (
    merged["financial_score"] * 0.35
    + merged["temporal_score"] * 0.25
    + merged["clinical_score"] * 0.3
    + merged["is_fraudulent"].fillna(0) * 0.3
)

merged["final_fraud_score"] = merged["anomaly_score"].clip(0, 1)

def band(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


merged["risk_band"] = merged["final_fraud_score"].apply(band)

# bucket used in risk distribution (can be adjusted)
merged["bucket"] = merged["risk_band"]

scored_df = merged.copy()

# ---------------------------------------------------------------------
# Pydantic models for API
# ---------------------------------------------------------------------
class RiskDistributionItem(BaseModel):
    label: str
    value: float


class TrendPoint(BaseModel):
    month: str
    value: float


class DrugItem(BaseModel):
    drug_name: str
    avg_fraud_score: float


class FraudReport(BaseModel):
    hospital_id: str
    year: int
    quarter: Optional[int] = None
    total_prescriptions: int
    high_risk_cases: int
    medium_risk_cases: int
    low_risk_cases: int
    controlled_drug_use: int
    active_alerts: int
    trend_last_3_months: List[TrendPoint]
    top_suspicious_drugs: List[DrugItem]
    risk_distribution: List[RiskDistributionItem]
    summary_text: str


class TopCase(BaseModel):
    prescription_id: str
    patient_id: str
    doctor_id: str
    doctor_specialty: Optional[str]
    drug_name: str
    quantity: float
    date: Optional[str]
    financial_score: float
    temporal_score: float
    clinical_score: float
    anomaly_score: float
    final_fraud_score: float
    risk_band: str
    recommended_action: str


class TopCasesResponse(BaseModel):
    hospital_id: str
    year: Optional[int]
    quarter: Optional[int]
    cases: List[TopCase]


# ---------------------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------------------
app = FastAPI(title="Hospital Prescription Fraud Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for hackathon / demo – tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def filter_hospital_year_quarter(
    df: pd.DataFrame, hospital_id: str, year: Optional[int], quarter: Optional[int]
) -> pd.DataFrame:
    sub = df[df["hospital_id"] == hospital_id].copy()
    if year is not None:
        sub = sub[sub["year"] == year]
    if quarter is not None:
        sub = sub[sub["quarter"] == quarter]
    return sub


def recommended_action_for_band(band: str) -> str:
    if band == "High":
        return "Needs investigation"
    if band == "Medium":
        return "Review required"
    return "Low risk"


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud backend is running"}


@app.get("/api/hospitals")
def list_hospitals():
    # Only expose id + name to frontend
    out = (
        hospitals_df[["hospital_id", "hospital_name"]]
        .drop_duplicates()
        .sort_values("hospital_id")
    )
    return out.to_dict(orient="records")


@app.get("/api/fraud-report", response_model=FraudReport)
def fraud_report(
    hospital_id: str,
    year: int,
    quarter: Optional[int] = None,
):
    sub = filter_hospital_year_quarter(scored_df, hospital_id, year, quarter)

    if sub.empty:
        # Still return a valid structure with zeros, so frontend doesn't break
        return FraudReport(
            hospital_id=hospital_id,
            year=year,
            quarter=quarter,
            total_prescriptions=0,
            high_risk_cases=0,
            medium_risk_cases=0,
            low_risk_cases=0,
            controlled_drug_use=0,
            active_alerts=0,
            trend_last_3_months=[
                TrendPoint(month="Month 1", value=0),
                TrendPoint(month="Month 2", value=0),
                TrendPoint(month="Month 3", value=0),
            ],
            top_suspicious_drugs=[],
            risk_distribution=[
                RiskDistributionItem(label="High", value=0),
                RiskDistributionItem(label="Medium", value=0),
                RiskDistributionItem(label="Low", value=0),
            ],
            summary_text="No prescriptions were found for this hospital and period.",
        )

    total_prescriptions = len(sub)

    high_risk_cases = (sub["risk_band"] == "High").sum()
    medium_risk_cases = (sub["risk_band"] == "Medium").sum()
    low_risk_cases = (sub["risk_band"] == "Low").sum()

    # controlled drug use
    controlled_drug_use = sub["is_controlled"].fillna(False).astype(int).sum()

    # "Active alerts" = all non-low
    active_alerts = int(high_risk_cases + medium_risk_cases)

    # Trend (last 3 months available)
    last_months = (
        sub.groupby(sub["date"].dt.to_period("M"))["final_fraud_score"]
        .mean()
        .sort_index()
    )

    last_months = last_months.tail(3)
    trend_points = []
    for i, (period, value) in enumerate(last_months.items(), start=1):
        trend_points.append(
            TrendPoint(month=f"Month {i}", value=round(float(value), 3))
        )

    # pad to 3 if fewer
    while len(trend_points) < 3:
        trend_points.insert(0, TrendPoint(month=f"Month {3-len(trend_points)}", value=0))

    # Top suspicious drugs by average fraud score
    drug_group = (
        sub.groupby("drug_name")["final_fraud_score"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
    )
    top_drugs = [
        DrugItem(drug_name=str(d), avg_fraud_score=round(float(score), 3))
        for d, score in drug_group.items()
    ]

    # Risk distribution
    dist = (
        sub.groupby("bucket")["risk_band"]
        .size()
        .reindex(["High", "Medium", "Low"], fill_value=0)
    )
    risk_distribution = [
        RiskDistributionItem(label=str(label), value=int(value))
        for label, value in dist.items()
    ]

    # Summary text (simple narrative)
    risk_level = "LOW"
    if high_risk_cases > 0:
        risk_level = "HIGH"
    elif medium_risk_cases > 0:
        risk_level = "MEDIUM"

    summary = (
        f"Current hospital risk level is classified as {risk_level} based on "
        f"{total_prescriptions} prescriptions in the selected period. "
        f"{high_risk_cases} prescriptions fall in the high-risk category and "
        f"{medium_risk_cases} in the medium-risk category."
    )

    return FraudReport(
        hospital_id=hospital_id,
        year=year,
        quarter=quarter,
        total_prescriptions=total_prescriptions,
        high_risk_cases=int(high_risk_cases),
        medium_risk_cases=int(medium_risk_cases),
        low_risk_cases=int(low_risk_cases),
        controlled_drug_use=int(controlled_drug_use),
        active_alerts=int(active_alerts),
        trend_last_3_months=trend_points,
        top_suspicious_drugs=top_drugs,
        risk_distribution=risk_distribution,
        summary_text=summary,
    )


@app.get("/api/top-cases", response_model=TopCasesResponse)
def top_cases(
    hospital_id: str,
    year: Optional[int] = None,
    quarter: Optional[int] = None,
    limit: int = 10,
):
    sub = filter_hospital_year_quarter(scored_df, hospital_id, year, quarter)

    if sub.empty:
        raise HTTPException(status_code=404, detail="No data for this filter")

    ranked = sub.sort_values("final_fraud_score", ascending=False).head(limit)

    cases: List[TopCase] = []
    for _, row in ranked.iterrows():
        cases.append(
            TopCase(
                prescription_id=str(row["prescription_id"]),
                patient_id=str(row["patient_id"]),
                doctor_id=str(row["doctor_id"]),
                doctor_specialty=row.get("specialty") or row.get("doctor_specialty"),
                drug_name=str(row["drug_name"]),
                quantity=float(row.get("quantity", 0) or 0),
                date=row["date"].strftime("%Y-%m-%d") if pd.notnull(row["date"]) else None,
                financial_score=round(float(row["financial_score"]), 3),
                temporal_score=round(float(row["temporal_score"]), 3),
                clinical_score=round(float(row["clinical_score"]), 3),
                anomaly_score=round(float(row["anomaly_score"]), 3),
                final_fraud_score=round(float(row["final_fraud_score"]), 3),
                risk_band=str(row["risk_band"]),
                recommended_action=recommended_action_for_band(str(row["risk_band"])),
            )
        )

    return TopCasesResponse(
        hospital_id=hospital_id,
        year=year,
        quarter=quarter,
        cases=cases,
    )
