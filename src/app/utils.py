"""
src/app/utils.py
─────────────────
Shared utilities: model loading, hierarchical decision engine, full scoring pipeline.

DECISION HIERARCHY (Point 1 + 2 from review):
  Step 1 — Fraud gate:        if fraud_score > FRAUD_DECLINE_THRESH → DECLINE_FRAUD
  Step 2 — Hard policy rules: if any hard rule fails → DECLINE_POLICY
  Step 3 — Credit risk:       if pd_pit > APPROVAL_THRESHOLD → DECLINE_CREDIT
  Step 4 — Refer band:        if pd_pit in (REFER_LOWER, APPROVAL_THRESHOLD] → REFER
  Step 5 — Approve:           all gates passed → APPROVE

This replaces the previous binary pd <= threshold logic and integrates fraud
as an upstream decision gate rather than a side module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import config

from src.features.imputation import impute
from src.features.interactions import add_interaction_features, compute_pit_to_ttc
from src.models.pd_model import pd_to_score, assign_risk_tier
from src.models.lgd_ead_model import (
    predict_lgd, compute_ead_term_loan, compute_ead_revolving,
    compute_expected_loss
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
    generate_credit_memo, generate_adverse_action_letter,
    generate_risk_summary, check_ollama_status
)

# ── Decision thresholds ────────────────────────────────────────────
APPROVAL_THRESHOLD   = 0.35   # PD above this = DECLINE_CREDIT
REFER_BAND_LOWER     = 0.28   # PD in (0.28, 0.35] = REFER (manual review)
FRAUD_DECLINE_THRESH = 0.65   # Fraud score above this = DECLINE_FRAUD

# Hard policy rules — apply before model scoring
# These represent lender policy floors, not model outputs.
HARD_POLICY = {
    "min_credit_score":  500,    # Hard bureau score floor
    "max_dti":           50.0,   # Max debt-to-income %
    "max_derog_marks":    3,     # Max derogatory marks (bankruptcies, charge-offs)
    "max_inquiries_6m":   8,     # Max hard inquiries in 6 months (loan stacking signal)
    "max_ltv_heloc":      0.90,  # Max LTV for HELOC
}

# ── Decision state metadata ────────────────────────────────────────
DECISION_CONFIG = {
    "APPROVE": {
        "badge_class": "approve-badge",
        "icon": "✅ APPROVED",
        "color": "#1a7a4a",
        "bg": "#d4edda",
        "description": "All credit, fraud, and policy criteria met.",
    },
    "REFER": {
        "badge_class": "refer-badge",
        "icon": "🔍 REFERRED — Manual Review Required",
        "color": "#7d5a00",
        "bg": "#fff3cd",
        "description": (
            "Application is in the manual review band. PD is elevated but below "
            "the automatic decline threshold. A credit analyst should review "
            "compensating factors before a final decision."
        ),
    },
    "DECLINE_CREDIT": {
        "badge_class": "decline-badge",
        "icon": "❌ DECLINED — Credit Risk",
        "color": "#c0392b",
        "bg": "#f8d7da",
        "description": "Probability of default exceeds the credit risk approval threshold.",
    },
    "DECLINE_POLICY": {
        "badge_class": "decline-badge",
        "icon": "❌ DECLINED — Policy Rule",
        "color": "#c0392b",
        "bg": "#f8d7da",
        "description": (
            "Application declined under a hard policy rule before credit model scoring. "
            "Policy rules are applied upstream of the model and cannot be overridden by "
            "compensating factors."
        ),
    },
    "DECLINE_FRAUD": {
        "badge_class": "decline-fraud-badge",
        "icon": "🚨 DECLINED — Fraud Alert",
        "color": "#6c0a0a",
        "bg": "#f5c6cb",
        "description": (
            "Application declined due to high fraud probability score. "
            "Account flagged for investigation. Do not disclose fraud reason to applicant "
            "per FCAC adverse action guidelines — use generic decline reason externally."
        ),
    },
}

TIER_COLORS = {
    "A": "#1a7a4a", "B": "#2d9e6b", "C": "#e6a817",
    "D": "#e07520", "E": "#c0392b",
}
TIER_LABELS = {
    "A": "A — Very Low Risk (720+)", "B": "B — Low Risk (680-719)",
    "C": "C — Moderate Risk (630-679)", "D": "D — Elevated Risk (580-629)",
    "E": "E — High Risk (<580)",
}
STAGE_COLORS = {1: "#1a7a4a", 2: "#e6a817", 3: "#c0392b"}
STAGE_LABELS = {
    1: "Stage 1 — Performing",
    2: "Stage 2 — SICR (Underperforming)",
    3: "Stage 3 — Credit Impaired",
}


# ── Model loading ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_models():
    """Load all model artifacts once and cache."""
    try:
        models = {
            "xgb":          joblib.load(config.XGB_PD_PATH),
            "scorecard":    joblib.load(config.SCORECARD_PATH),
            "lgd":          joblib.load(config.LGD_MODEL_PATH),
            "shap":         joblib.load("models/shap_explainer.pkl"),
            "feat_cfg":     joblib.load("models/feature_config.pkl"),
            "medians":      joblib.load("models/imputation_medians.pkl"),
            "lgd_features": joblib.load("models/lgd_features.pkl"),
            "loaded":       True,
        }
        # Separate product models — load if available (v1.1+), else None
        for key, path in [
            ("xgb_unsecured", "models/xgb_pd_unsecured.pkl"),
            ("xgb_secured",   "models/xgb_pd_secured.pkl"),
        ]:
            try:
                models[key] = joblib.load(path)
            except FileNotFoundError:
                models[key] = None

        # Fraud model — load if available
        try:
            models["fraud"] = joblib.load("models/fraud_model.pkl")
            models["fraud_features"] = joblib.load("models/fraud_features.pkl")
        except FileNotFoundError:
            models["fraud"] = None
            models["fraud_features"] = None

        models["has_separate_models"] = (
            models["xgb_unsecured"] is not None and
            models["xgb_secured"] is not None
        )
        return models

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return {"loaded": False}


@st.cache_data(show_spinner=False)
def load_portfolio():
    """Load test portfolio with regulatory metrics."""
    try:
        return pd.read_parquet("data/processed/portfolio_regulatory.parquet")
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_reports():
    """Load all Phase 4/6 report CSVs."""
    reports = {}
    for key, path in [
        ("model_comparison",           "reports/phase4/model_comparison.csv"),
        ("model_comparison_segmented", "reports/phase4/model_comparison_segmented.csv"),
        ("vintage",                    "reports/phase4/vintage_curves.csv"),
        ("csi",                        "reports/phase4/csi_monitoring.csv"),
        ("stress",                     "reports/phase6/stress_test_results.csv"),
        ("ecl",                        "reports/phase6/ecl_summary.csv"),
        ("capital",                    "reports/phase6/capital_summary.csv"),
    ]:
        try:
            reports[key] = pd.read_csv(path)
        except Exception:
            reports[key] = pd.DataFrame()
    return reports


# ── Tab-specific analytical helpers (used by Tabs 3, 4, 5) ─────────

def compute_roc_curve(y_true, y_pred, n_points: int = 200):
    """
    Compute ROC curve points without depending on sklearn at call time.
    Returns (fpr, tpr, thresholds).
    """
    import numpy as np
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    thr = np.linspace(0.0, 1.0, n_points)
    tpr = np.empty_like(thr)
    fpr = np.empty_like(thr)
    p = max(y_true.sum(), 1)
    n = max((1 - y_true).sum(), 1)
    for i, t in enumerate(thr):
        pred_pos = (y_pred >= t).astype(int)
        tpr[i] = ((pred_pos == 1) & (y_true == 1)).sum() / p
        fpr[i] = ((pred_pos == 1) & (y_true == 0)).sum() / n
    return fpr, tpr, thr


def compute_ks_curve(y_true, y_pred):
    """
    Returns (score_bins, cum_bad_rate, cum_good_rate, ks_max, ks_score_at_max).
    """
    import numpy as np
    import pandas as pd
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    df = pd.DataFrame({"y": y_true, "p": y_pred}).sort_values("p").reset_index(drop=True)
    df["cum_bad"]  = (df["y"]     ).cumsum() / max(df["y"].sum(), 1)
    df["cum_good"] = (1 - df["y"] ).cumsum() / max((1 - df["y"]).sum(), 1)
    df["ks_gap"]   = (df["cum_good"] - df["cum_bad"]).abs()
    ks_max = float(df["ks_gap"].max())
    score_at_max = float(df.loc[df["ks_gap"].idxmax(), "p"])
    return df["p"].values, df["cum_bad"].values, df["cum_good"].values, ks_max, score_at_max


def confusion_at_threshold(y_true, y_pred, threshold: float) -> dict:
    """Confusion matrix + precision/recall/F1/FPR at a given PD cutoff."""
    import numpy as np
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    pos = (y_pred > threshold).astype(int)    # predicted decline
    tp = int(((pos == 1) & (y_true == 1)).sum())
    fp = int(((pos == 1) & (y_true == 0)).sum())
    tn = int(((pos == 0) & (y_true == 0)).sum())
    fn = int(((pos == 0) & (y_true == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    fpr  = fp / max(fp + tn, 1)
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "fpr":       round(fpr,  4),
        "n":         int(len(y_true)),
    }


def compute_lift_gains(y_true, y_pred, n_deciles: int = 10):
    """Lift and cumulative gains by score decile (decile 1 = highest PD)."""
    import numpy as np
    import pandas as pd
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    df = pd.DataFrame({"y": y_true, "p": y_pred})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, n_deciles, labels=False, duplicates="drop") + 1
    baseline = df["y"].mean()
    agg = df.groupby("decile").agg(
        n=("y", "size"),
        defaults=("y", "sum"),
        default_rate=("y", "mean"),
    ).reset_index()
    agg["lift"] = agg["default_rate"] / max(baseline, 1e-9)
    agg["cum_defaults_pct"] = agg["defaults"].cumsum() / max(agg["defaults"].sum(), 1)
    agg["cum_pop_pct"] = agg["n"].cumsum() / agg["n"].sum()
    return agg, baseline


def compute_tier_pds_from_portfolio(portfolio) -> dict:
    """
    Portfolio-derived mid-PD per risk tier.
    Returns dict: {"A": 0.038, "B": 0.095, ...}
    """
    import pandas as pd
    if portfolio is None or len(portfolio) == 0 or "risk_tier" not in portfolio.columns:
        # Fallback to hardcoded defaults if no portfolio data
        return {"A": 0.04, "B": 0.09, "C": 0.18, "D": 0.30, "E": 0.48}
    out = {}
    for tier in ["A", "B", "C", "D", "E"]:
        sub = portfolio[portfolio["risk_tier"] == tier]
        if len(sub) > 0 and "pd_score" in sub.columns:
            out[tier] = float(sub["pd_score"].median())
        else:
            out[tier] = {"A": 0.04, "B": 0.09, "C": 0.18,
                         "D": 0.30, "E": 0.48}[tier]
    return out


def compute_profit_curve(portfolio, cof: float = 0.055, opex: float = 0.015,
                          roe: float = 0.020, rate_cap: float = 0.35) -> "pd.DataFrame":
    """
    Sweep approval thresholds (PD cutoffs) and compute:
      - approval rate
      - expected interest revenue (under risk-based pricing up to cap)
      - expected credit loss
      - net profit
      - default rate on approved book
    """
    import numpy as np
    import pandas as pd
    if portfolio is None or len(portfolio) == 0:
        return pd.DataFrame()
    req = {"pd_score", "lgd_estimate", "ead_estimate"}
    if not req.issubset(portfolio.columns):
        return pd.DataFrame()

    p = portfolio[["pd_score", "lgd_estimate", "ead_estimate"]].dropna().copy()
    # Build risk-based rate: CoF + EL_rate + OpEx + ROE, capped at 35%
    p["el_rate"]  = p["pd_score"] * p["lgd_estimate"]
    p["rate_uncapped"] = cof + p["el_rate"] + opex + roe
    p["rate"] = p["rate_uncapped"].clip(upper=rate_cap)
    # Annual expected interest revenue per $ of EAD
    p["interest_rev"] = p["rate"] * p["ead_estimate"]
    # Expected credit loss per $ of EAD
    p["ecl_dollar"]   = p["el_rate"] * p["ead_estimate"]

    thresholds = np.arange(0.05, 0.71, 0.02)
    rows = []
    n_total = len(p)
    for t in thresholds:
        approved = p[p["pd_score"] <= t]
        if len(approved) == 0:
            continue
        rev  = approved["interest_rev"].sum()
        ecl  = approved["ecl_dollar"].sum()
        rows.append({
            "threshold":    round(float(t), 3),
            "approval_rate": len(approved) / n_total,
            "n_approved":   len(approved),
            "expected_revenue": float(rev),
            "expected_loss":    float(ecl),
            "net_profit":       float(rev - ecl),
            "avg_pd_approved":  float(approved["pd_score"].mean()),
        })
    return pd.DataFrame(rows)


# ── Hard policy rule evaluation ────────────────────────────────────

def evaluate_policy_rules(applicant: dict) -> tuple[bool, list[str]]:
    """
    Evaluate hard policy rules upstream of model scoring.
    Returns (all_pass: bool, failures: list[str]).
    Policy rules represent lender policy floors — not model outputs.
    They apply regardless of PD and cannot be overridden.
    """
    failures = []
    cs    = float(applicant.get("credit_score", 700))
    dti   = float(applicant.get("dti", 25))
    derog = float(applicant.get("num_derogatory_marks", 0))
    inq   = float(applicant.get("num_inquiries_last_6m", 0))
    pt    = int(applicant.get("product_type", 0))
    ltv   = float(applicant.get("ltv_ratio") or 0)

    if cs < HARD_POLICY["min_credit_score"]:
        failures.append(
            f"Credit score {cs:.0f} is below the minimum policy floor of "
            f"{HARD_POLICY['min_credit_score']}. This is an absolute rule — "
            f"no compensating factors apply below this threshold."
        )
    if dti > HARD_POLICY["max_dti"]:
        failures.append(
            f"Debt-to-income ratio {dti:.1f}% exceeds maximum policy limit of "
            f"{HARD_POLICY['max_dti']:.0f}%. Applicant is over-indebted relative "
            f"to income."
        )
    if derog >= HARD_POLICY["max_derog_marks"]:
        failures.append(
            f"{derog:.0f} derogatory marks on file. Policy maximum is "
            f"{HARD_POLICY['max_derog_marks']}. Derogatory marks include "
            f"bankruptcies, charge-offs, and collections."
        )
    if inq > HARD_POLICY["max_inquiries_6m"]:
        failures.append(
            f"{inq:.0f} hard inquiries in the last 6 months exceeds the policy "
            f"maximum of {HARD_POLICY['max_inquiries_6m']}. This is a loan "
            f"stacking / velocity fraud signal."
        )
    if pt == 1 and ltv > HARD_POLICY["max_ltv_heloc"]:
        failures.append(
            f"HELOC LTV ratio {ltv:.0%} exceeds maximum policy limit of "
            f"{HARD_POLICY['max_ltv_heloc']:.0%}. Insufficient equity cushion."
        )
    return len(failures) == 0, failures


# ── Fraud scoring for single application ──────────────────────────

def score_fraud_single(models: dict, df_features: pd.DataFrame,
                       applicant: dict) -> dict:
    """
    Score a single application through the fraud model.
    Returns fraud probability, alert tier, and key flags.
    """
    if models.get("fraud") is None:
        return {
            "fraud_score": 0.0,
            "alert_tier": "NOT_SCORED",
            "fpd_risk": False,
            "high_velocity": False,
            "available": False,
        }

    try:
        fraud_feats = models.get("fraud_features", [])
        # Build fraud feature row from available applicant data
        fraud_row = {
            "credit_score":           applicant.get("credit_score", 660),
            "pd_score":               0.20,  # Will be filled after PD scoring
            "fpd_risk_score":         0.05,
            "fpd_risk_flag":          False,
            "high_velocity_flag":     applicant.get("num_inquiries_last_6m", 0) > 5,
            "velocity_score":         min(applicant.get("num_inquiries_last_6m", 0) / 10, 1),
            "high_loan_to_income":    (applicant.get("loan_amount", 10000) /
                                       max(applicant.get("annual_income", 50000), 1)) > 0.5,
            "income_loan_ratio":      (max(applicant.get("annual_income", 50000), 1) /
                                       max(applicant.get("loan_amount", 10000), 1)),
            "short_tenure_flag":      applicant.get("employment_length_years", 3) < 1,
            "score_income_inconsistency": 0.0,
            "synthetic_id_score":     0.10,
            "synthetic_id_risk_flag": False,
            "address_mismatch_flag":  False,
            "doc_verify_score":       0.80,
            "fraud_feature_score":    0.05,
            "lgd_estimate":           0.65,
            "product_type":           applicant.get("product_type", 0),
            "multi_app_flag":         False,
        }
        # Build X from available fraud features
        available = [f for f in fraud_feats if f in fraud_row]
        if not available:
            return {"fraud_score": 0.0, "alert_tier": "LOW",
                    "fpd_risk": False, "high_velocity": False, "available": False}

        X_fraud = pd.DataFrame([{f: fraud_row[f] for f in available}])
        fraud_score = float(models["fraud"].predict_proba(X_fraud)[:, 1][0])

        # Alert tier
        if fraud_score >= 0.50:
            tier = "CONFIRMED"
        elif fraud_score >= 0.25:
            tier = "HIGH"
        elif fraud_score >= 0.10:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        return {
            "fraud_score":    fraud_score,
            "alert_tier":     tier,
            "fpd_risk":       fraud_row["fpd_risk_flag"],
            "high_velocity":  fraud_row["high_velocity_flag"],
            "available":      True,
        }
    except Exception:
        return {"fraud_score": 0.0, "alert_tier": "LOW",
                "fpd_risk": False, "high_velocity": False, "available": False}


# ── Hierarchical decision engine ───────────────────────────────────

def make_decision(pd_pit: float, applicant: dict,
                  fraud_result: dict) -> tuple[str, str, list[str]]:
    """
    Apply the full decision hierarchy.
    Returns (decision_code, reason, policy_failures).

    Decision codes:
      APPROVE        — All gates passed
      REFER          — PD in manual review band (28-35%)
      DECLINE_CREDIT — PD exceeds approval threshold
      DECLINE_POLICY — Hard policy rule failure
      DECLINE_FRAUD  — Fraud score exceeds fraud gate threshold
    """
    # Step 1: Fraud gate
    fraud_score = fraud_result.get("fraud_score", 0.0)
    if fraud_result.get("available") and fraud_score > FRAUD_DECLINE_THRESH:
        return (
            "DECLINE_FRAUD",
            f"Fraud probability {fraud_score:.1%} exceeds the fraud gate threshold "
            f"({FRAUD_DECLINE_THRESH:.0%}). Account flagged for investigation.",
            []
        )

    # Step 2: Hard policy rules
    policy_pass, policy_failures = evaluate_policy_rules(applicant)
    if not policy_pass:
        return (
            "DECLINE_POLICY",
            f"Application failed {len(policy_failures)} hard policy rule(s).",
            policy_failures
        )

    # Step 3: Credit risk threshold
    if pd_pit > APPROVAL_THRESHOLD:
        return (
            "DECLINE_CREDIT",
            f"Probability of default {pd_pit:.1%} exceeds approval threshold "
            f"{APPROVAL_THRESHOLD:.0%}.",
            []
        )

    # Step 4: Refer band
    if pd_pit > REFER_BAND_LOWER:
        return (
            "REFER",
            f"PD {pd_pit:.1%} is in the manual review band "
            f"({REFER_BAND_LOWER:.0%}–{APPROVAL_THRESHOLD:.0%}). "
            f"Analyst review required for final decision.",
            []
        )

    # Step 5: Approve
    return (
        "APPROVE",
        "All fraud, policy, and credit criteria met.",
        []
    )


# ── Main scoring pipeline ──────────────────────────────────────────

def score_applicant(models: dict, applicant: dict) -> dict:
    """
    Full scoring pipeline for one applicant.
    Applies the hierarchical decision engine:
      fraud gate → hard policy → credit model → refer band → approve
    """
    RAW_FEATURES = models["feat_cfg"]["raw_features"]
    LGD_FEATURES = models["lgd_features"]
    product_int  = int(applicant.get("product_type", 0))

    # Build feature row
    row = {
        "loan_amount":               applicant.get("loan_amount", 10000),
        "annual_income":             applicant.get("annual_income", 60000),
        "dti":                       applicant.get("dti", 25.0),
        "credit_score":              applicant.get("credit_score", 660),
        "employment_length_years":   applicant.get("employment_length_years", 3),
        "product_type":              product_int,
        "loan_term_months":          applicant.get("loan_term_months", 36),
        "num_derogatory_marks":      applicant.get("num_derogatory_marks", 0),
        "total_accounts":            applicant.get("total_accounts", 10),
        "alt_data_score":            applicant.get("alt_data_score", 50),
        "ltv_ratio":                 applicant.get("ltv_ratio"),
        "thin_file_flag":            applicant.get("thin_file_flag", False),
        "months_since_oldest_trade": applicant.get("months_since_oldest_trade", 60),
        "num_inquiries_last_6m":     applicant.get("num_inquiries_last_6m", 1),
        "credit_utilization":        applicant.get("credit_utilization", 35),
        "months_since_recent_delinquency": None,
        "months_since_recent_trade": None,
        "external_risk_estimate":    None,
        "pct_trades_never_delinquent": None,
        "num_high_utilization_trades": None,
        "has_synthetic_features":    False,
        "default_flag":              0,
        "origination_year":          2025,
        "data_source":               "streamlit",
        "num_open_accounts":         8.0,
    }

    df = pd.DataFrame([row])

    # Loan term standardisation — product proxy mitigation
    # ──────────────────────────────────────────────────────
    # loan_term_months acts as a product-type proxy in the dual-product model.
    # Training data only contained 36 and 60-month unsecured loans and 120-month
    # HELOC records (imputed to 36). Values outside this range push the model
    # into interpolation territory and dominate SHAP incorrectly.
    #
    # Unsecured: map to nearest training value (36 or 60 months only).
    #   - 12, 24, 36 → 36 (short/medium term)
    #   - 48, 60, 120 → 60 (long term)
    # HELOC: standardise to 36 (the training imputation value for HELOC).
    #
    # This is a documented interim mitigation — separate product models (v1.1)
    # will remove this proxy entirely.
    if product_int == 1:
        df["loan_term_months"] = 36.0   # HELOC — always use training value
    else:
        # Unsecured — snap to nearest training value
        raw_term = float(df["loan_term_months"].iloc[0])
        df["loan_term_months"] = 60.0 if raw_term >= 48 else 36.0

    df, _ = impute(df, medians=models["medians"], fit=False)
    df    = add_interaction_features(df)
    X     = df[RAW_FEATURES].fillna(0)
    X     = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Fraud scoring (before credit model — upstream gate)
    fraud_result = score_fraud_single(models, X, applicant)

    # PD — use separate product model if available (v1.1+), else combined
    if models.get("has_separate_models"):
        xgb_model = (
            models["xgb_secured"] if product_int == 1
            else models["xgb_unsecured"]
        )
        model_used = "segmented"
    else:
        xgb_model  = models["xgb"]
        model_used = "combined"

    pd_pit  = float(xgb_model.predict_proba(X)[:, 1][0])
    pd_ttc  = float(compute_pit_to_ttc(np.array([pd_pit]))[0])
    cr_sc   = int(pd_to_score(np.array([pd_pit]))[0])
    tier    = str(assign_risk_tier(np.array([cr_sc]))[0])

    # Hierarchical decision
    decision, decision_reason, policy_failures = make_decision(
        pd_pit, applicant, fraud_result
    )

    # LGD / EAD / EL
    lgd      = float(predict_lgd(models["lgd"], df, LGD_FEATURES)[0])
    loan_amt = float(applicant.get("loan_amount", 10000))
    term     = float(applicant.get("loan_term_months", 36))
    ead      = float(
        compute_ead_revolving(np.array([loan_amt]), np.array([product_int]))[0]
        if product_int == 1
        else compute_ead_term_loan(np.array([loan_amt]),
                                    loan_term_months=np.array([term]))[0]
    )
    el = float(compute_expected_loss(
        np.array([pd_pit]), np.array([lgd]), np.array([ead])
    )[0])

    # Regulatory
    irb = compute_irb_capital(
        np.array([pd_ttc]), np.array([lgd]), np.array([ead])
    )
    stage = int(assign_ifrs9_stage(
        np.array([pd_pit]), np.array([pd_pit * 0.8]),
        np.array([cr_sc]),  np.array([cr_sc + 15])
    )[0])
    ecl_val = float(compute_ecl(
        np.array([pd_pit]), np.array([lgd]), np.array([ead]),
        np.array([stage]), np.array([term])
    )[0])

    # Risk-based rate
    el_rate    = pd_pit * lgd
    collateral = (
        -0.005 * max(0, (0.80 - (applicant.get("ltv_ratio") or 0)) / 0.10)
        if product_int == 1 else 0
    )
    raw_rate = 0.055 + el_rate + 0.015 + 0.020 + collateral
    rate     = min(raw_rate, 0.35)

    # SHAP — filter loan_term_months for HELOC display
    explanation   = get_local_explanation(models["xgb"], X.iloc[0], RAW_FEATURES)
    formatted     = format_local_explanation(explanation, top_n=5)
    all_factors   = (formatted["top_risk_drivers"] +
                     formatted["top_protective_factors"])

    display_factors = [
        f for f in all_factors
        if not (product_int == 1 and "loan_term" in f.get("feature", "").lower())
    ]

    adverse_reasons = generate_adverse_action_reasons(explanation, top_n=3)
    top_factor = display_factors[0]["feature"] if display_factors else "credit_score"

    # Counterfactual (declined or referred)
    cf = None
    if decision in ("DECLINE_CREDIT", "REFER"):
        try:
            cf = find_counterfactual(
                models["xgb"], X, RAW_FEATURES,
                approval_threshold=APPROVAL_THRESHOLD
            )
        except Exception:
            cf = None

    return {
        # Decision
        "decision":          decision,
        "decision_reason":   decision_reason,
        "policy_failures":   policy_failures,
        "model_used":        model_used,

        # Fraud
        "fraud_score":       fraud_result.get("fraud_score", 0.0),
        "fraud_alert_tier":  fraud_result.get("alert_tier", "NOT_SCORED"),
        "fraud_available":   fraud_result.get("available", False),

        # PD
        "pd_pit":            pd_pit,
        "pd_ttc":            pd_ttc,
        "credit_score":      cr_sc,
        "risk_tier":         tier,

        # Loss
        "lgd":               lgd,
        "ead":               ead,
        "el":                el,
        "el_rate":           el_rate,

        # Regulatory
        "rwa":               float(irb["rwa"][0]),
        "min_capital":       float(irb["min_capital"][0]),
        "ifrs9_stage":       stage,
        "ecl":               ecl_val,
        "risk_based_rate":   rate,

        # Explanation
        "all_factors":       all_factors,
        "display_factors":   display_factors,
        "adverse_reasons":   adverse_reasons,
        "top_factor":        top_factor,
        "counterfactual":    cf,

        # Raw data
        "X":                 X,
        "applicant":         applicant,
    }
