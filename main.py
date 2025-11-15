# main.py
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------------------------
# Paths & data loading
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


def _read_csv(name: str) -> pd.DataFrame:
    """
    Helper to read a CSV file from the project folder.
    We use UTF-8 so Arabic hospital names work correctly.
    """
    path = BASE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, encoding="utf-8")


# Load all CSVs once at startup
transactions_df = _read_csv("Transactions_Clean.csv")
hospitals_df = _read_csv("Hospitals.csv")
drugs_df = _read_csv("Drugs.csv")
doctors_df = _read_csv("Doctors.csv")
patients_df = _read_csv("Patients.csv")

# -------------------------------------------------------------------
# Clean / normalize columns (year, quarter, month)
# -------------------------------------------------------------------
def _parse_int_like(val):
    """
    Turn '2025', 'Q2', 'q3', 'Month 4', '4' -> 2025, 2, 3, 4, 4
    Returns np.nan if it can't parse.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    # remove common prefixes
    for prefix in ("q", "quarter", "month", "m"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    # keep only digits
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return np.nan
    try:
        return int(digits)
    except ValueError:
        return np.nan


# year should already be numeric, but make sure
transactions_df["year"] = transactions_df["year"].apply(_parse_int_like).astype("Int64")

# quarter might be 'Q1', 'Q2', etc.
if "quarter" in transactions_df.columns:
    transactions_df["quarter"] = (
        transactions_df["quarter"].apply(_parse_int_like).astype("Int64")
    )

# month can be 1-12 or strings like 'Month 1'
if "month" in transactions_df.columns:
    transactions_df["month"] = transactions_df["month"].apply(_parse_int_like).astype(
        "Int64"
    )

# If is_controlled is 0/1, coerce to bool safely
if "is_controlled" in drugs_df.columns:
    drugs_df["is_controlled"] = drugs_df["is_controlled"].fillna(0).astype(bool)

# -------------------------------------------------------------------
# Scoring helpers
# -------------------------------------------------------------------
def score_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a subset of transactions, merge with drugs and compute:
    - financial_score   (price vs typical price)
    - volume_score      (quantity vs median for that drug)
    - control_score     (bonus if controlled substance)
    - final_fraud_score (combined)
    - risk_band         (Low / Medium / High)
    """
    merged = df.merge(
        drugs_df[["drug_name", "is_controlled", "typical_price"]],
        on="drug_name",
        how="left",
    )

    # Typical price: prefer Drugs.csv; otherwise median unit_price per drug
    merged["typical_price"] = merged["typical_price"].fillna(
        merged.groupby("drug_name")["unit_price"].transform("median")
    )
    merged["typical_price"] = merged["typical_price"].fillna(merged["unit_price"])

    # Financial score – how expensive vs typical
    price_ratio = merged["unit_price"] / merged["typical_price"].replace(0, np.nan)
    price_ratio = price_ratio.fillna(1.0)
    merged["financial_score"] = np.clip((price_ratio - 1.0) / 2.0, 0, 1.0)  # 1 when >=3x

    # Volume score – quantity vs median for that drug
    median_qty = merged.groupby("drug_name")["quantity"].transform("median")
    qty_ratio = merged["quantity"] / median_qty.replace(0, np.nan)
    qty_ratio = qty_ratio.fillna(1.0)
    merged["volume_score"] = np.clip((qty_ratio - 1.0) / 4.0, 0, 1.0)  # 1 when >=5x

    # Control score – controlled substances get a bonus
    merged["control_score"] = np.where(
        merged["is_controlled"].fillna(False), 0.3, 0.0
    )

    # Final score and risk band
    merged["final_fraud_score"] = np.clip(
        merged["financial_score"] + merged["volume_score"] + merged["control_score"],
        0,
        1.0,
    )

    conditions = [
        merged["final_fraud_score"] >= 0.66,
        merged["final_fraud_score"] >= 0.33,
    ]
    choices = ["High", "Medium"]
    merged["risk_band"] = np.select(conditions, choices, default="Low")

    return merged


def filter_scored(
    hospital_id: str, year: int, quarter: Optional[int]
) -> pd.DataFrame:
    """
    Filter the big table for a hospital/year[/quarter] and apply scoring.
    """
    df = transactions_df[transactions_df["hospital_id"] == hospital_id]
    df = df[df["year"] == year]
    if quarter is not None and "quarter" in df.columns:
        df = df[df["quarter"] == quarter]

    if df.empty:
        return df

    return score_transactions(df.copy())


def build_fraud_report(
    hospital_id: str, year: int, quarter: Optional[int]
) -> Dict:
    scored = filter_scored(hospital_id, year, quarter)
    if scored.empty:
        raise HTTPException(
            status_code=404,
            detail="No transactions found for this hospital / year / quarter.",
        )

    total = int(len(scored))
    band_counts = scored["risk_band"].value_counts().to_dict()
    high = int(band_counts.get("High", 0))
    med = int(band_counts.get("Medium", 0))
    low = int(band_counts.get("Low", 0))
    controlled = int(scored["is_controlled"].fillna(False).sum())
    active_alerts = high  # simple rule: all high-risk cases are alerts

    # Fraud trend – last 3 months available in this subset
    if "month" in scored.columns:
        months = sorted([m for m in scored["month"].dropna().unique()])
    else:
        months = []

    last3 = months[-3:]
    trend = []
    for m in last3:
        sub = scored[scored["month"] == m]
        value = int((sub["risk_band"] == "High").sum())
        trend.append({"month": f"Month {m}", "value": value})

    # Top 3 suspicious drugs by mean fraud score
    drug_scores = (
        scored.groupby("drug_name")["final_fraud_score"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
    )
    top_drugs = [
        {"drug_name": name, "avg_fraud_score": round(float(val), 3)}
        for name, val in drug_scores.items()
    ]

    # Risk distribution as an OBJECT (for frontend pie chart)
    risk_distribution = {
        "High": high,
        "Medium": med,
        "Low": low,
    }

    # Overall hospital risk level
    high_ratio = high / total if total > 0 else 0
    if high_ratio >= 0.25:
        risk_level = "HIGH"
    elif high_ratio >= 0.10:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    hospital_row = hospitals_df.loc[
        hospitals_df["hospital_id"] == hospital_id
    ].iloc[0]

    quarter_label = None if quarter is None else f"Q{quarter}"

    # Human-readable summary for the UI
    if quarter is None:
        period_label = f"{year} (full year)"
    else:
        period_label = f"{year} Q{quarter}"

    summary = (
        f"{hospital_row['hospital_name']} had {total} prescriptions in {period_label}, "
        f"with {high} high-risk, {med} medium-risk, and {low} low-risk cases. "
        f"{controlled} prescriptions involved controlled drugs."
    )

    return {
        "hospital_id": hospital_id,
        "hospital_name": hospital_row["hospital_name"],
        "year": int(year),
        "quarter": quarter_label,
        "hospital_risk_level": risk_level,          # renamed for frontend
        "total_prescriptions": total,
        "high_risk_cases": high,
        "medium_risk_cases": med,
        "low_risk_cases": low,
        "controlled_drug_use": controlled,
        "active_alerts": active_alerts,
        "summary": summary,                         # added summary
        "trend_last_3_months": trend,               # renamed for frontend
        "top_suspicious_drugs": top_drugs,
        "risk_distribution": risk_distribution,     # now {High, Medium, Low}
    }


def build_top_cases(
    hospital_id: str, year: int, quarter: Optional[int], limit: int = 5
) -> List[Dict]:
    scored = filter_scored(hospital_id, year, quarter)
    if scored.empty:
        raise HTTPException(
            status_code=404,
            detail="No transactions found for this hospital / year / quarter.",
        )

    scored = scored.sort_values("final_fraud_score", ascending=False).head(limit)

    cases: List[Dict] = []
    for _, row in scored.iterrows():
        if row["risk_band"] == "High":
            action = "Review immediately – high-risk prescription."
        elif row["risk_band"] == "Medium":
            action = "Flag for clinical review."
        else:
            action = "Monitor periodically."

        cases.append(
            {
                "prescription_id": row["prescription_id"],
                "patient_id": row["patient_id"],
                "doctor_id": row["doctor_id"],
                "drug_name": row["drug_name"],
                "quantity": int(row["quantity"]),
                "date": row["date"],
                "financial_score": round(float(row["financial_score"]), 3),
                "volume_score": round(float(row["volume_score"]), 3),
                "control_score": round(float(row["control_score"]), 3),
                "final_fraud_score": round(float(row["final_fraud_score"]), 3),
                "risk_band": row["risk_band"],
                "recommended_action": action,
            }
        )

    return cases


# -------------------------------------------------------------------
# FastAPI app + CORS
# -------------------------------------------------------------------
app = FastAPI(title="Prescription Fraud Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development / Netlify – you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud backend is running"}


@app.get("/api/hospitals")
def list_hospitals():
    """
    Returns all hospitals as {id, name, label}.
    Label is 'name (H001)' – good for dropdowns.
    """
    items = []
    for _, row in hospitals_df.sort_values("hospital_id").iterrows():
        items.append(
            {
                "id": row["hospital_id"],
                "name": row["hospital_name"],
                "label": f"{row['hospital_name']} ({row['hospital_id']})",
            }
        )
    return {"hospitals": items}


@app.get("/api/fraud-report")
def fraud_report(
    hospital_id: str,
    year: int,
    quarter: Optional[int] = Query(
        None, ge=1, le=4, description="Quarter 1–4. Omit for full year."
    ),
):
    """
    Main dashboard endpoint.
    Example:
    /api/fraud-report?hospital_id=H001&year=2025
    /api/fraud-report?hospital_id=H001&year=2025&quarter=4
    """
    if hospital_id not in set(hospitals_df["hospital_id"]):
        raise HTTPException(status_code=404, detail="Unknown hospital_id")

    report = build_fraud_report(hospital_id, year, quarter)
    return report


@app.get("/api/top-cases")
def top_cases(
    hospital_id: str,
    year: int,
    quarter: Optional[int] = Query(
        None, ge=1, le=4, description="Quarter 1–4. Omit for full year."
    ),
    limit: int = Query(5, ge=1, le=50),
):
    """
    Returns top-N suspicious prescriptions for a hospital / year / quarter.
    """
    if hospital_id not in set(hospitals_df["hospital_id"]):
        raise HTTPException(status_code=404, detail="Unknown hospital_id")

    return {
        "hospital_id": hospital_id,
        "year": year,
        "quarter": None if quarter is None else f"Q{quarter}",
        "cases": build_top_cases(hospital_id, year, quarter, limit),
    }
