# main.py
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------
def load_csv_safe(name: str, encodings=("utf-8", "cp1256", "latin-1")) -> pd.DataFrame:
    path = BASE_DIR / name
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # fallback
    if last_err:
        print(f"[WARN] Could not decode {name} cleanly, using latin-1:", last_err)
    return pd.read_csv(path, encoding="latin-1")


transactions_df = load_csv_safe("Transactions_Clean.csv", encodings=("latin-1", "utf-8"))
patients_df = load_csv_safe("Patients.csv")
hospitals_df = load_csv_safe("Hospitals.csv")
drugs_df = load_csv_safe("Drugs.csv")
doctors_df = load_csv_safe("Doctors.csv")

# ---------------------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------------------
# Ensure date / year / quarter exist
if "date" in transactions_df.columns:
    transactions_df["date"] = pd.to_datetime(transactions_df["date"], errors="coerce")
else:
    transactions_df["date"] = pd.NaT

transactions_df["year"] = transactions_df["date"].dt.year
transactions_df["quarter"] = transactions_df["date"].dt.quarter

# If no hospital_id column, try a reasonable fallback
if "hospital_id" not in transactions_df.columns:
    raise RuntimeError("Transactions_Clean.csv must contain 'hospital_id' column")

# Merge into one big dataframe for scoring
merged = (
    transactions_df
    .merge(patients_df, on="patient_id", how="left", suffixes=("", "_patient"))
    .merge(doctors_df, on="doctor_id", how="left", suffixes=("", "_doctor"))
    .merge(hospitals_df, on="hospital_id", how="left", suffixes=("", "_hospital"))
    .merge(drugs_df, on="drug_name", how="left", suffixes=("", "_drug"))
)

# ---------------------------------------------------------------------
# Simple fraud scoring (lightweight & robust)
# ---------------------------------------------------------------------
def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def compute_score(row) -> float:
    """
    Very simple heuristic that only uses columns that usually exist.
    This is NOT for real use – just to drive the dashboard safely.
    """
    qty = safe_float(row.get("quantity", 0))
    total_price = safe_float(row.get("total_price", row.get("unit_price", 0)))
    # normalize quantity
    qty_score = min(1.0, qty / 10.0)
    price_score = min(1.0, total_price / 500.0)

    # If we happen to have a label column, use it
    label_cols = ["is_fraudulent", "fraud_label", "fraud"]
    label_score = 0.0
    for col in label_cols:
        if col in row and safe_float(row[col]) > 0:
            label_score = 1.0
            break

    return max(0.0, min(1.0, 0.4 * qty_score + 0.4 * price_score + 0.4 * label_score))


merged["final_fraud_score"] = merged.apply(compute_score, axis=1)

def band(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"

merged["risk_band"] = merged["final_fraud_score"].apply(band)
merged["bucket"] = merged["risk_band"]
scored_df = merged.copy()

# ---------------------------------------------------------------------
# Pydantic models
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
    trend_last_3_months: list[TrendPoint]
    top_suspicious_drugs: list[DrugItem]
    risk_distribution: list[RiskDistributionItem]
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
    cases: list[TopCase]


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(title="Fraud Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud backend is running"}


@app.get("/api/hospitals")
def list_hospitals():
    # Try Arabic name column first if it exists
    name_col = "hospital_name_ar" if "hospital_name_ar" in hospitals_df.columns else (
        "hospital_name" if "hospital_name" in hospitals_df.columns else None
    )

    cols = ["hospital_id"]
    if name_col:
        cols.append(name_col)

    df = hospitals_df[cols].drop_duplicates().sort_values("hospital_id")

    records = []
    for _, row in df.iterrows():
        item = {"hospital_id": row["hospital_id"]}
        if name_col:
            item["hospital_name"] = str(row[name_col])
        records.append(item)
    return records


def filter_hospital_year_quarter(
    df: pd.DataFrame, hospital_id: str, year: Optional[int], quarter: Optional[int]
) -> pd.DataFrame:
    sub = df[df["hospital_id"] == hospital_id].copy()
    if year is not None:
        sub = sub[sub["year"] == year]
    if quarter is not None:
        sub = sub[sub["quarter"] == quarter]
    return sub


def recommended_action(b: str) -> str:
    if b == "High":
        return "Needs investigation"
    if b == "Medium":
        return "Review required"
    return "Low risk"


@app.get("/api/fraud-report", response_model=FraudReport)
def fraud_report(hospital_id: str, year: int, quarter: Optional[int] = None):
    sub = filter_hospital_year_quarter(scored_df, hospital_id, year, quarter)

    if sub.empty:
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
                TrendPoint(month="Month 1", value=0.0),
                TrendPoint(month="Month 2", value=0.0),
                TrendPoint(month="Month 3", value=0.0),
            ],
            top_suspicious_drugs=[],
            risk_distribution=[
                RiskDistributionItem(label="High", value=0),
                RiskDistributionItem(label="Medium", value=0),
                RiskDistributionItem(label="Low", value=0),
            ],
            summary_text="No prescriptions found for this filter.",
        )

    total_prescriptions = len(sub)
    high_risk_cases = int((sub["risk_band"] == "High").sum())
    medium_risk_cases = int((sub["risk_band"] == "Medium").sum())
    low_risk_cases = int((sub["risk_band"] == "Low").sum())

    # controlled drug use if the column exists
    if "is_controlled" in sub.columns:
        controlled_drug_use = int(sub["is_controlled"].fillna(0).astype(int).sum())
    else:
        controlled_drug_use = 0

    active_alerts = high_risk_cases + medium_risk_cases

    # trend – last 3 months
    dates = sub["date"]
    if dates.notna().any():
        trend_raw = (
            sub.groupby(sub["date"].dt.to_period("M"))["final_fraud_score"]
            .mean()
            .sort_index()
            .tail(3)
        )
        trend_points = [
            TrendPoint(month=f"Month {i+1}", value=float(v))
            for i, (_, v) in enumerate(trend_raw.items())
        ]
    else:
        trend_points = [
            TrendPoint(month="Month 1", value=0.0),
            TrendPoint(month="Month 2", value=0.0),
            TrendPoint(month="Month 3", value=0.0),
        ]

    while len(trend_points) < 3:
        trend_points.insert(
            0, TrendPoint(month=f"Month {3 - len(trend_points)}", value=0.0)
        )

    # top drugs
    if "drug_name" in sub.columns:
        drug_group = (
            sub.groupby("drug_name")["final_fraud_score"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
        )
        top_drugs = [
            DrugItem(drug_name=str(d), avg_fraud_score=float(s))
            for d, s in drug_group.items()
        ]
    else:
        top_drugs = []

    # risk distribution
    dist = (
        sub.groupby("bucket")["risk_band"]
        .size()
        .reindex(["High", "Medium", "Low"], fill_value=0)
    )
    risk_distribution = [
        RiskDistributionItem(label=str(lbl), value=int(val))
        for lbl, val in dist.items()
    ]

    # summary
    risk_level = "LOW"
    if high_risk_cases > 0:
        risk_level = "HIGH"
    elif medium_risk_cases > 0:
        risk_level = "MEDIUM"

    summary_text = (
        f"Current hospital risk level is {risk_level} with "
        f"{high_risk_cases} high-risk and {medium_risk_cases} medium-risk prescriptions "
        f"out of {total_prescriptions} total in the selected period."
    )

    return FraudReport(
        hospital_id=hospital_id,
        year=year,
        quarter=quarter,
        total_prescriptions=int(total_prescriptions),
        high_risk_cases=high_risk_cases,
        medium_risk_cases=medium_risk_cases,
        low_risk_cases=low_risk_cases,
        controlled_drug_use=controlled_drug_use,
        active_alerts=int(active_alerts),
        trend_last_3_months=trend_points,
        top_suspicious_drugs=top_drugs,
        risk_distribution=risk_distribution,
        summary_text=summary_text,
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
                prescription_id=str(row.get("prescription_id", "")),
                patient_id=str(row.get("patient_id", "")),
                doctor_id=str(row.get("doctor_id", "")),
                doctor_specialty=row.get("specialty") or row.get("doctor_specialty"),
                drug_name=str(row.get("drug_name", "")),
                quantity=safe_float(row.get("quantity", 0.0)),
                date=row["date"].strftime("%Y-%m-%d")
                if pd.notna(row.get("date"))
                else None,
                financial_score=0.0,  # we didn't compute separate components
                temporal_score=0.0,
                clinical_score=0.0,
                anomaly_score=float(row["final_fraud_score"]),
                final_fraud_score=float(row["final_fraud_score"]),
                risk_band=str(row["risk_band"]),
                recommended_action=recommended_action(str(row["risk_band"])),
            )
        )

    return TopCasesResponse(
        hospital_id=hospital_id,
        year=year,
        quarter=quarter,
        cases=cases,
    )
