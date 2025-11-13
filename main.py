import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.ensemble import IsolationForest


# ---------- Helpers ----------

def month_to_quarter(m: int) -> int:
    if m in (1, 2, 3):
        return 1
    if m in (4, 5, 6):
        return 2
    if m in (7, 8, 9):
        return 3
    return 4


def safe_float(v):
    """Convert to JSON-safe float (NaN/inf -> None)."""
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        if np.isinf(v):
            return None
    except Exception:
        pass
    return float(v)


def has_chronic_condition(row, keywords):
    conds = str(row.get("chronic_conditions", "")).lower()
    return any(k.lower() in conds for k in keywords)


def risk_band(score: float) -> str:
    if score >= 0.8:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


def recommended_action(score: float) -> str:
    if score >= 0.8:
        return "Needs investigation"
    if score >= 0.5:
        return "Review required"
    return "Low risk"


# ---------- Data loading & feature engineering ----------

def build_dataframe() -> pd.DataFrame:
    # Load CSVs
    base_path = "data"
    transactions = pd.read_csv(f"{base_path}/Transactions_Clean.csv")
    patients     = pd.read_csv(f"{base_path}/Patients.csv")
    hospitals    = pd.read_csv(f"{base_path}/Hospitals.csv")
    drugs        = pd.read_csv(f"{base_path}/Drugs.csv")
    doctors      = pd.read_csv(f"{base_path}/Doctors.csv")

    # Hospital key
    if "pharmacy_id" in transactions.columns:
        transactions["hospital_key"] = transactions["pharmacy_id"]
    elif "hospital_id" in transactions.columns:
        transactions["hospital_key"] = transactions["hospital_id"]
    else:
        raise ValueError(
            "Transactions must contain 'pharmacy_id' or 'hospital_id'. "
            f"Found: {transactions.columns.tolist()}"
        )

    # Date
    transactions["date"] = pd.to_datetime(transactions["date"])

    # Merge to big DF
    df = transactions.merge(drugs, on="drug_name", how="left")
    df = df.merge(patients, on="patient_id", how="left")
    df = df.merge(
        hospitals,
        left_on="hospital_key",
        right_on="hospital_id",
        how="left",
        suffixes=("", "_hosp"),
    )
    df = df.merge(
        doctors,
        on="doctor_id",
        how="left",
        suffixes=("", "_doc"),
    )

    # Time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["month"].apply(month_to_quarter)

    # Controlled flag
    try:
        df["controlled_flag"] = df["is_controlled"].astype(int)
    except Exception:
        df["controlled_flag"] = (
            df["is_controlled"].astype(str).str.lower()
            .isin(["1", "true", "yes", "y"])
            .astype(int)
        )

    df["qty_log"] = np.log1p(df["quantity"])

    # ---------- Financial Agent ----------
    def compute_financial_score(row):
        tp = row.get("typical_price", np.nan)
        up = row.get("unit_price", np.nan)
        if pd.isna(tp) or tp <= 0 or pd.isna(up) or up <= 0:
            return 0.0
        ratio = up / tp
        if ratio <= 1.2:
            return 0.1 * (ratio / 1.2)
        elif ratio <= 2.0:
            return 0.4 + 0.4 * ((ratio - 1.2) / 0.8)
        else:
            return 0.8 + min((ratio - 2.0) / 3, 0.2)

    df["financial_score"] = df.apply(compute_financial_score, axis=1)

    # ---------- Temporal Agent ----------
    df = df.sort_values(by=["patient_id", "drug_name", "date"])
    df["days_since_last_drug"] = (
        df.groupby(["patient_id", "drug_name"])["date"].diff().dt.days
    )

    def compute_temporal_score(row):
        d = row["days_since_last_drug"]
        if pd.isna(d):
            return 0.0
        is_ctrl = int(row["controlled_flag"])
        if is_ctrl == 1:
            if d < 7:
                return 1.0
            elif d < 14:
                return 0.7
            elif d < 30:
                return 0.4
            else:
                return 0.1
        else:
            if d < 7:
                return 0.6
            elif d < 14:
                return 0.3
            else:
                return 0.1

    df["temporal_score"] = df.apply(compute_temporal_score, axis=1)

    # ---------- Clinical Agent ----------
    strong_opioids = [
        "morphine",
        "oxycodone",
        "fentanyl",
        "tramadol",
        "hydrocodone",
        "codeine",
    ]

    def compute_clinical_score(row):
        drug = str(row["drug_name"]).lower()
        chronic = str(row.get("chronic_conditions", "")).strip()
        is_ctrl = bool(row.get("is_controlled", False))

        if chronic == "":
            return 0.9 if is_ctrl else 0.3

        if has_chronic_condition(row, ["chronic pain", "cancer", "oncology"]):
            return 0.2 if any(s in drug for s in strong_opioids) else 0.1

        if has_chronic_condition(row, ["asthma"]):
            return 0.1 if "salbutamol" in drug else 0.3

        return 0.7 if is_ctrl else 0.2

    df["clinical_score"] = df.apply(compute_clinical_score, axis=1)

    # ---------- Anomaly Agent ----------
    anomaly_features = ["quantity", "unit_price", "controlled_flag", "qty_log"]
    X = df[anomaly_features].fillna(0)

    iso = IsolationForest(contamination=0.1, random_state=42)
    iso_labels = iso.fit_predict(X)
    df["anomaly_score"] = np.where(iso_labels == -1, 1.0, 0.0)

    # ---------- Meta-Learner ----------
    df["final_fraud_score"] = (
        0.30 * df["financial_score"]
        + 0.25 * df["temporal_score"]
        + 0.25 * df["clinical_score"]
        + 0.20 * df["anomaly_score"]
    )

    df["risk_band"] = df["final_fraud_score"].apply(risk_band)
    df["recommended_action"] = df["final_fraud_score"].apply(recommended_action)

    return df


# Build DF once at startup
df = build_dataframe()


# ---------- Hospital-level analytics ----------

def analyze_hospital(hospital_id: str, year: int | None = None, quarter: int | None = None):
    sub = df[df["hospital_key"] == hospital_id].copy()

    if year is not None:
        sub = sub[sub["year"] == year]
    if quarter is not None:
        sub = sub[sub["quarter"] == quarter]

    if sub.empty:
        return None

    total_prescriptions = len(sub)
    high_risk_cases = (sub["risk_band"] == "High").sum()
    medium_risk_cases = (sub["risk_band"] == "Medium").sum()
    low_risk_cases = (sub["risk_band"] == "Low").sum()
    controlled_count = sub["controlled_flag"].sum()
    refill_alerts = (sub["temporal_score"] >= 0.7).sum()
    doctor_risk_count = sub[sub["risk_band"] == "High"]["doctor_id"].nunique()
    avg_fraud_score = sub["final_fraud_score"].mean()
    hospital_risk_level = risk_band(avg_fraud_score)

    top_drugs = (
        sub.groupby("drug_name")["final_fraud_score"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    summary = (
        f"For hospital {hospital_id}, the system analyzed {total_prescriptions} prescriptions. "
        f"There are {high_risk_cases} high-risk and {medium_risk_cases} medium-risk cases, "
        f"with notable usage of controlled drugs. The overall hospital risk level is "
        f"{hospital_risk_level}."
    )

    return {
        "hospital_id": hospital_id,
        "year": year,
        "quarter": quarter,
        "total_prescriptions": int(total_prescriptions),
        "high_risk_cases": int(high_risk_cases),
        "medium_risk_cases": int(medium_risk_cases),
        "low_risk_cases": int(low_risk_cases),
        "controlled_drug_prescriptions": int(controlled_count),
        "refill_alerts": int(refill_alerts),
        "doctor_risk_count": int(doctor_risk_count),
        "average_fraud_score": safe_float(avg_fraud_score),
        "hospital_risk_level": hospital_risk_level,
        "top_suspicious_drugs": top_drugs,
        "summary_text": summary,
    }


# ---------- FastAPI app ----------

app = FastAPI(title="Hospital Pharmacy Fraud Detection API")

# CORS so your Netlify frontend can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://your-app.netlify.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/fraud-report")
def fraud_report(hospital_id: str, year: int | None = None, quarter: int | None = None):
    try:
        report = analyze_hospital(hospital_id, year=year, quarter=quarter)
        if report is None:
            return JSONResponse({"error": "No data for this filter"}, status_code=404)
        return report
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/top-cases")
def top_cases(hospital_id: str, year: int | None = None, quarter: int | None = None, limit: int = 10):
    try:
        sub = df[df["hospital_key"] == hospital_id].copy()

        if year is not None:
            sub = sub[sub["year"] == year]
        if quarter is not None:
            sub = sub[sub["quarter"] == quarter]

        if sub.empty:
            return JSONResponse({"error": "No data for this filter"}, status_code=404)

        sub = sub.sort_values(by="final_fraud_score", ascending=False).head(limit)
        sub = sub.replace({np.nan: None})  # make JSON-safe

        cases = []
        for _, row in sub.iterrows():
            cases.append({
                "prescription_id": row["prescription_id"],
                "patient_id": row["patient_id"],
                "doctor_id": row["doctor_id"],
                "doctor_specialty": row.get("specialty", None),
                "drug_name": row["drug_name"],
                "quantity": int(row["quantity"]) if row["quantity"] is not None else None,
                "date": row["date"].strftime("%Y-%m-%d") if row["date"] is not None else None,
                "financial_score": safe_float(row.get("financial_score")),
                "temporal_score": safe_float(row.get("temporal_score")),
                "clinical_score": safe_float(row.get("clinical_score")),
                "anomaly_score": safe_float(row.get("anomaly_score")),
                "final_fraud_score": safe_float(row.get("final_fraud_score")),
                "risk_band": row["risk_band"],
                "recommended_action": row["recommended_action"],
            })

        return {
            "hospital_id": hospital_id,
            "year": year,
            "quarter": quarter,
            "cases": cases,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# For local testing only:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
