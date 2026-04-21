"""
src/api/main.py
────────────────
FastAPI credit scoring API — LOS-ready endpoint with hierarchical decision engine.

DECISION HIERARCHY:
  POST /score applies: fraud gate → hard policy → credit model → refer → approve
  Returns decision code: APPROVE | REFER | DECLINE_CREDIT | DECLINE_POLICY | DECLINE_FRAUD

ENDPOINTS:
  POST /score          — Score a single applicant (full pipeline)
  POST /score/batch    — Score multiple applicants (CSV upload)
  GET  /health         — Health check + model status
  GET  /model/info     — Model metadata, features, decision thresholds
  GET  /ollama/status  — Check Ollama LLM availability

USAGE:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib
import logging
import time
import io

import config
from src.features.imputation import impute
from src.features.interactions import add_interaction_features, compute_pit_to_ttc
from src.models.pd_model import pd_to_score, assign_risk_tier
from src.models.lgd_ead_model import (
    predict_lgd, compute_ead_term_loan,
    compute_ead_revolving, compute_expected_loss
)
from src.models.regulatory import (
    compute_irb_capital, assign_ifrs9_stage, compute_ecl
)
from src.explainability.shap_explainer import (
    get_local_explanation, format_local_explanation,
    generate_adverse_action_reasons
)
from src.explainability.counterfactuals import find_counterfactual
from src.llm.ollama_client import (
    generate_credit_memo, generate_adverse_action_letter, check_ollama_status
)

# Import decision constants from utils
from src.app.utils import (
    APPROVAL_THRESHOLD, REFER_BAND_LOWER, FRAUD_DECLINE_THRESH,
    HARD_POLICY, make_decision, evaluate_policy_rules, score_fraud_single
)

log = logging.getLogger(__name__)

# ── Load model artifacts ─────────────────────────────────────────
print("Loading model artifacts ...")
try:
    XGB_MODEL       = joblib.load(config.XGB_PD_PATH)
    SCORECARD       = joblib.load(config.SCORECARD_PATH)
    LGD_MODEL       = joblib.load(config.LGD_MODEL_PATH)
    SHAP_EXPLAINER  = joblib.load("models/shap_explainer.pkl")
    FEAT_CONFIG     = joblib.load("models/feature_config.pkl")
    MEDIANS         = joblib.load("models/imputation_medians.pkl")
    LGD_FEATURES    = joblib.load("models/lgd_features.pkl")
    MODELS_LOADED   = True
    print("  All model artifacts loaded.")
except Exception as e:
    log.error(f"Model loading failed: {e}")
    MODELS_LOADED = False
    print(f"  WARNING: {e}")

# Separate product models (v1.1+)
XGB_UNSECURED = XGB_SECURED = None
HAS_SEPARATE_MODELS = False
try:
    XGB_UNSECURED = joblib.load("models/xgb_pd_unsecured.pkl")
    XGB_SECURED   = joblib.load("models/xgb_pd_secured.pkl")
    HAS_SEPARATE_MODELS = True
    print("  Separate product models loaded.")
except FileNotFoundError:
    print("  Separate product models not found — using combined model.")

# Fraud model
FRAUD_MODEL = FRAUD_FEATURES = None
try:
    FRAUD_MODEL    = joblib.load("models/fraud_model.pkl")
    FRAUD_FEATURES = joblib.load("models/fraud_features.pkl")
    print("  Fraud model loaded.")
except FileNotFoundError:
    print("  Fraud model not found.")

RAW_FEATURES = FEAT_CONFIG["raw_features"] if MODELS_LOADED else []

# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "LOS-ready credit scoring endpoint with hierarchical decision engine. "
        "Applies fraud gate → hard policy rules → credit model → refer band → approve. "
        "Returns decision code, fraud score, PD, SHAP explanations, and LLM credit memo."
    ),
    version="1.0.0 — Preliminary Release",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────

class ApplicantRequest(BaseModel):
    loan_amount:             float = Field(..., gt=0, le=500_000)
    annual_income:           float = Field(..., gt=0)
    dti:                     float = Field(..., ge=0, le=100)
    credit_score:            float = Field(..., ge=300, le=850)
    employment_length_years: float = Field(default=3.0, ge=0, le=40)
    product_type:            int   = Field(default=0, description="0=unsecured, 1=HELOC")
    loan_term_months:        Optional[float] = Field(default=36.0)
    credit_utilization:      Optional[float] = Field(default=None, ge=0, le=100)
    num_derogatory_marks:    Optional[float] = Field(default=0.0, ge=0)
    num_inquiries_last_6m:   Optional[float] = Field(default=0.0, ge=0)
    total_accounts:          Optional[float] = Field(default=10.0, ge=0)
    ltv_ratio:               Optional[float] = Field(default=None, ge=0, le=1)
    alt_data_score:          Optional[float] = Field(default=50.0, ge=0, le=100)
    thin_file_flag:          Optional[bool]  = Field(default=False)
    applicant_name:          Optional[str]   = Field(default="Applicant")
    generate_memo:           Optional[bool]  = Field(default=True)

    class Config:
        json_schema_extra = {"example": {
            "loan_amount": 15000, "annual_income": 65000, "dti": 28.5,
            "credit_score": 680, "employment_length_years": 5, "product_type": 0,
        }}


class SHAPFactor(BaseModel):
    feature: str; value: float; shap_impact: float; direction: str


class ScoreResponse(BaseModel):
    # Decision
    decision:          str   = Field(description="APPROVE|REFER|DECLINE_CREDIT|DECLINE_POLICY|DECLINE_FRAUD")
    decision_reason:   str
    policy_failures:   List[str]
    approval_threshold:float
    refer_band_lower:  float
    model_version:     str

    # Fraud
    fraud_score:       float
    fraud_alert_tier:  str

    # PD
    pd_score:          float
    pd_ttc:            float
    risk_score:        int   = Field(description="Internal PDO risk score 300-850. Not a bureau score.")
    risk_tier:         str

    # Loss
    lgd:               float; ead: float; expected_loss: float

    # Regulatory
    rwa:               float; min_capital: float
    ifrs9_stage:       int;   ecl: float
    risk_based_rate:   float

    # Explanation
    shap_factors:      List[SHAPFactor]
    adverse_reasons:   List[str]

    # Memo
    credit_memo:       Optional[str]
    latency_ms:        float


# ── Scoring function ─────────────────────────────────────────────

def _score(req: ApplicantRequest) -> ScoreResponse:
    if not MODELS_LOADED:
        raise HTTPException(503, "Models not loaded. Run build.py first.")

    t0 = time.time()
    applicant = req.dict()
    product_int = int(req.product_type)

    row = {
        "loan_amount": req.loan_amount, "annual_income": req.annual_income,
        "dti": req.dti, "credit_score": req.credit_score,
        "employment_length_years": req.employment_length_years,
        "product_type": product_int, "loan_term_months": req.loan_term_months or 36,
        "num_derogatory_marks": req.num_derogatory_marks or 0,
        "total_accounts": req.total_accounts or 10,
        "num_inquiries_last_6m": req.num_inquiries_last_6m or 0,
        "credit_utilization": req.credit_utilization or 35,
        "alt_data_score": req.alt_data_score or 50,
        "ltv_ratio": req.ltv_ratio, "thin_file_flag": req.thin_file_flag or False,
        "months_since_oldest_trade": 60, "months_since_recent_delinquency": None,
        "months_since_recent_trade": None, "external_risk_estimate": None,
        "pct_trades_never_delinquent": None, "num_high_utilization_trades": None,
        "has_synthetic_features": False, "default_flag": 0,
        "origination_year": 2025, "data_source": "api", "num_open_accounts": 8.0,
    }

    df = pd.DataFrame([row])
    if product_int == 1:
        df["loan_term_months"] = 36.0  # HELOC proxy mitigation

    df, _ = impute(df, medians=MEDIANS, fit=False)
    df    = add_interaction_features(df)
    X     = df[RAW_FEATURES].fillna(0)
    X     = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Fraud scoring
    _models = {
        "fraud": FRAUD_MODEL, "fraud_features": FRAUD_FEATURES,
        "xgb": XGB_MODEL, "has_separate_models": HAS_SEPARATE_MODELS,
        "xgb_unsecured": XGB_UNSECURED, "xgb_secured": XGB_SECURED,
    }
    fraud_result = score_fraud_single(_models, X, applicant)

    # PD
    xgb = (XGB_SECURED if product_int==1 and HAS_SEPARATE_MODELS
           else XGB_UNSECURED if product_int==0 and HAS_SEPARATE_MODELS
           else XGB_MODEL)
    model_ver = "segmented_v1.1" if HAS_SEPARATE_MODELS else "combined_v1.0"

    pd_pit = float(xgb.predict_proba(X)[:, 1][0])
    pd_ttc = float(compute_pit_to_ttc(np.array([pd_pit]))[0])
    cr_sc  = int(pd_to_score(np.array([pd_pit]))[0])
    tier   = str(assign_risk_tier(np.array([cr_sc]))[0])

    # Hierarchical decision
    decision, decision_reason, policy_failures = make_decision(
        pd_pit, applicant, fraud_result
    )

    # LGD / EAD / EL
    lgd = float(predict_lgd(LGD_MODEL, df, LGD_FEATURES)[0])
    ead = float(
        compute_ead_revolving(np.array([req.loan_amount]), np.array([product_int]))[0]
        if product_int == 1
        else compute_ead_term_loan(np.array([req.loan_amount]),
                                    loan_term_months=np.array([req.loan_term_months or 36]))[0]
    )
    el = float(compute_expected_loss(np.array([pd_pit]), np.array([lgd]), np.array([ead]))[0])

    irb = compute_irb_capital(np.array([pd_ttc]), np.array([lgd]), np.array([ead]))
    stage = int(assign_ifrs9_stage(
        np.array([pd_pit]), np.array([pd_pit*0.8]),
        np.array([cr_sc]), np.array([cr_sc+15])
    )[0])
    ecl_val = float(compute_ecl(
        np.array([pd_pit]), np.array([lgd]), np.array([ead]),
        np.array([stage]), np.array([req.loan_term_months or 36])
    )[0])

    rate = min(0.055 + pd_pit*lgd + 0.015 + 0.020, 0.35)

    # SHAP
    explanation = get_local_explanation(XGB_MODEL, X.iloc[0], RAW_FEATURES)
    formatted   = format_local_explanation(explanation, top_n=5)
    all_factors = formatted["top_risk_drivers"] + formatted["top_protective_factors"]
    adverse     = generate_adverse_action_reasons(explanation, top_n=3)

    # Memo
    memo = None
    if req.generate_memo and decision != "DECLINE_FRAUD":
        memo = generate_credit_memo(
            applicant=applicant, pd_score=pd_pit, credit_score=cr_sc, risk_tier=tier,
            lgd=lgd, ead=ead, expected_loss=el,
            shap_factors=all_factors, decision=decision, product_type=product_int
        )

    return ScoreResponse(
        decision=decision, decision_reason=decision_reason,
        policy_failures=policy_failures,
        approval_threshold=APPROVAL_THRESHOLD, refer_band_lower=REFER_BAND_LOWER,
        model_version=model_ver,
        fraud_score=fraud_result.get("fraud_score", 0.0),
        fraud_alert_tier=fraud_result.get("alert_tier", "NOT_SCORED"),
        pd_score=pd_pit, pd_ttc=pd_ttc, risk_score=cr_sc, risk_tier=tier,
        lgd=lgd, ead=ead, expected_loss=el,
        rwa=float(irb["rwa"][0]), min_capital=float(irb["min_capital"][0]),
        ifrs9_stage=stage, ecl=ecl_val, risk_based_rate=rate,
        shap_factors=[SHAPFactor(**f) for f in all_factors if all(k in f for k in ["feature","value","shap_impact","direction"])],
        adverse_reasons=adverse,
        credit_memo=memo,
        latency_ms=round((time.time()-t0)*1000, 1),
    )


# ── Endpoints ───────────────────────────────────────────────────

@app.post("/score", response_model=ScoreResponse)
def score_single(req: ApplicantRequest):
    """Score a single applicant through the full hierarchical decision engine."""
    return _score(req)


@app.get("/health")
def health():
    return {
        "status": "healthy" if MODELS_LOADED else "degraded",
        "models_loaded": MODELS_LOADED,
        "separate_product_models": HAS_SEPARATE_MODELS,
        "fraud_model_loaded": FRAUD_MODEL is not None,
        "decision_thresholds": {
            "approval": APPROVAL_THRESHOLD,
            "refer_lower": REFER_BAND_LOWER,
            "fraud_decline": FRAUD_DECLINE_THRESH,
        },
        "hard_policy": HARD_POLICY,
        "version": "1.0.0-preliminary",
    }


@app.get("/model/info")
def model_info():
    return {
        "model": "XGBoost + Platt Calibration",
        "version": "combined_v1.0 (separate product models in v1.1)",
        "features": RAW_FEATURES,
        "calibration": "Platt scaling — mean PD gap +0.0001",
        "auc_oot": 0.6769,
        "ks_oot": 0.2532,
        "gini_oot": 0.3537,
        "training_data": "LendingClub 2007-2015 (315,895 rows)",
        "test_data": "LendingClub + HELOC 2016-2018 (194,564 rows) — OOT",
        "known_limitations": [
            "AUC 0.68 due to dual-product architecture (loan_term_months as product proxy)",
            "Separate product models (v1.1) expected to push AUC to 0.75-0.82",
            "Fraud labels are synthetic — not from confirmed investigations",
            "IFRS 9 origination PD is grade-based proxy, not actual LOS records",
        ],
        "decision_hierarchy": [
            "1. Fraud gate (fraud_score > 65% → DECLINE_FRAUD)",
            "2. Hard policy rules (credit floor, DTI cap, derog max, inquiry cap)",
            "3. Credit model (PD > 35% → DECLINE_CREDIT)",
            "4. Refer band (PD 28-35% → REFER)",
            "5. Approve (all gates pass → APPROVE)",
        ],
    }


@app.get("/ollama/status")
def ollama_status():
    return check_ollama_status()
