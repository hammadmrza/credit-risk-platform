"""
src/explainability/shap_explainer.py
──────────────────────────────────────
SHAP (SHapley Additive exPlanations) module.

SHAP is the industry standard for explaining machine learning
model predictions. It answers two questions:

GLOBAL: Which features matter most across the entire portfolio?
  → Used by: model validators, regulators, risk committees
  → Output: beeswarm plot, feature importance bar chart

LOCAL: Why did THIS specific applicant get THIS score?
  → Used by: loan officers, adverse action letters, credit memos
  → Output: waterfall chart, top positive/negative factors

SHAP is grounded in cooperative game theory (Shapley values from
economics). For each prediction, it calculates how much each
feature contributed — positively or negatively — to the final
score, compared to the average prediction across all borrowers.

REGULATORY RELEVANCE:
  OSFI E-23 requires that model decisions can be explained.
  ECOA/Reg B requires specific adverse action reasons.
  SHAP provides the quantitative basis for both.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
import logging
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


def compute_shap_values(model,
                         X: pd.DataFrame,
                         sample_size: int = 500) -> Tuple:
    """
    Compute SHAP values for a fitted XGBoost model.

    Handles PlattCalibratedModel wrapper — extracts the base
    XGBoost model for TreeExplainer compatibility.
    SHAP values are computed on the base model (ranking order
    is preserved; only the probability scale changes with calibration).
    """
    import shap

    # Unwrap PlattCalibratedModel to get base XGBoost for TreeExplainer
    base_model = getattr(model, "base", model)

    log.info(f"Computing SHAP values (sample={min(sample_size, len(X))}) ...")

    if len(X) > sample_size:
        idx = np.random.RandomState(42).choice(
            len(X), size=sample_size, replace=False
        )
        X_sample = X.iloc[idx].copy()
    else:
        X_sample = X.copy()

    explainer   = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_sample)

    log.info(f"  SHAP values computed: shape={shap_values.shape}")
    return explainer, shap_values, X_sample


def get_global_importance(shap_values: np.ndarray,
                           feature_names: List[str]) -> pd.DataFrame:
    """
    Compute global feature importance from SHAP values.
    Importance = mean(|SHAP value|) across all samples.

    Args:
        shap_values: SHAP value matrix (n_samples × n_features).
        feature_names: Feature column names.

    Returns:
        DataFrame sorted by importance descending.
    """
    importance = pd.DataFrame({
        "feature":    feature_names,
        "importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance["rank"]     = range(1, len(importance) + 1)
    importance["pct_total"] = (importance["importance"] /
                                importance["importance"].sum() * 100).round(1)
    return importance


def get_local_explanation(model,
                           X_single: pd.Series,
                           feature_names: List[str],
                           base_value: float = None) -> pd.DataFrame:
    """
    Compute SHAP explanation for a single applicant.

    Args:
        model: Fitted model.
        X_single: Single row as a Series (one applicant).
        feature_names: Feature names.
        base_value: Average model prediction (baseline).

    Returns:
        DataFrame of feature contributions, sorted by absolute impact.
    """
    import shap
    base_model = getattr(model, "base", model)
    try:
        base_model.base_score = 0.5
    except Exception:
        pass
    explainer   = shap.TreeExplainer(base_model)
    X_df        = pd.DataFrame([X_single.values], columns=X_single.index)
    shap_vals   = explainer.shap_values(X_df)[0]
    base        = base_value or float(explainer.expected_value)

    explanation = pd.DataFrame({
        "feature":      feature_names,
        "feature_value": X_single[feature_names].values,
        "shap_value":   shap_vals,
        "abs_impact":   np.abs(shap_vals),
        "direction":    np.where(shap_vals > 0, "increases_risk",
                                 "decreases_risk"),
    }).sort_values("abs_impact", ascending=False).reset_index(drop=True)

    explanation["base_prediction"] = base
    return explanation


def format_local_explanation(explanation: pd.DataFrame,
                              top_n: int = 5) -> Dict:
    """
    Format local SHAP explanation into structured dict for
    Ollama prompts and Streamlit display.

    Args:
        explanation: Output of get_local_explanation().
        top_n: Number of top factors to return.

    Returns:
        Dict with top_risk_drivers, top_protective_factors, summary.
    """
    risk_drivers = explanation[
        explanation["direction"] == "increases_risk"
    ].head(top_n)

    protective = explanation[
        explanation["direction"] == "decreases_risk"
    ].head(top_n)

    def format_factor(row):
        return {
            "feature":     row["feature"],
            "value":       round(float(row["feature_value"]), 3),
            "shap_impact": round(float(row["shap_value"]), 4),
            "direction":   row["direction"],
        }

    return {
        "top_risk_drivers":       [format_factor(r)
                                    for _, r in risk_drivers.iterrows()],
        "top_protective_factors": [format_factor(r)
                                    for _, r in protective.iterrows()],
        "base_prediction":        float(explanation["base_prediction"].iloc[0]),
        "n_features_analysed":    len(explanation),
    }


def generate_adverse_action_reasons(explanation: pd.DataFrame,
                                     top_n: int = 3) -> List[str]:
    """
    Generate ECOA/Regulation B compliant adverse action reasons
    from SHAP values.

    The Equal Credit Opportunity Act and Regulation B require lenders
    to provide specific, principal reasons for adverse action.
    SHAP provides the quantitative basis for those reasons.

    Args:
        explanation: Output of get_local_explanation().
        top_n: Number of adverse action reasons (typically 3-4).

    Returns:
        List of plain-language adverse action reason strings.
    """
    # Standard ECOA reason code mapping
    REASON_MAP = {
        "credit_score":                   "Insufficient credit score",
        "dti":                            "Debt obligations too high relative to income",
        "num_derogatory_marks":           "Derogatory credit history on file",
        "credit_utilization":             "Excessive revolving credit utilization",
        "num_inquiries_last_6m":          "Elevated number of recent credit applications",
        "months_since_recent_delinquency":"Recent delinquency on credit accounts",
        "employment_length_years":        "Insufficient length of employment",
        "annual_income":                  "Income insufficient relative to requested amount",
        "ltv_ratio":                      "Loan-to-value ratio exceeds guidelines",
        "total_accounts":                 "Insufficient depth of credit history",
        "loan_amount":                    "Requested loan amount exceeds risk guidelines",
        "months_since_oldest_trade":      "Limited length of credit history",
        "alt_data_score":                 "Alternative payment history insufficient",
        "num_high_utilization_trades":    "Multiple accounts near credit limits",
        "external_risk_estimate":         "External risk indicators unfavourable",
        "pct_trades_never_delinquent":    "Payment history shows prior delinquencies",
        "ltv_x_product":                  "Collateral coverage insufficient for product",
        "score_x_product":                "Credit profile insufficient for product type",
        # Interaction features — map to plain English, never show raw feature name
        "dti_x_emp_risk":                 "Combined debt load and employment risk elevated",
        "ads_x_thin_file":                "Limited credit history with insufficient alt data",
        "ltv_x_product":                  "Collateral value insufficient for product",
        "score_x_product":                "Credit score insufficient for product type",
    }

    # Features that should never appear in adverse action reasons
    # (internal model features, interaction terms, or non-bureau signals)
    ALWAYS_SUPPRESS = {
        "dti_x_emp_risk",          # Interaction feature — not explainable to consumer
        "ads_x_thin_file",         # Synthetic feature — not a real bureau signal
        "score_x_product",         # Product interaction — not meaningful to applicant
        "ltv_x_product",           # Handled separately via LTV reason
        "loan_term_months",        # Product proxy — not a creditworthiness signal
        "has_synthetic_features",  # Internal flag
        "product_type",            # Not an adverse reason
    }

    # Value thresholds below which a feature is NOT a meaningful adverse reason.
    # Even if SHAP impact is slightly positive, citing a feature with a normal
    # value as an adverse reason would be misleading and legally indefensible.
    # Under ECOA, adverse reasons must reflect genuine risk factors.
    SUPPRESS_IF_BELOW = {
        "num_inquiries_last_6m":      3,    # 1-2 inquiries = normal, not adverse
        "num_derogatory_marks":       1,    # 0 derog marks = no adverse history
        "credit_utilization":        50,    # < 50% utilization = not adverse
        "dti":                       36,    # < 36% DTI = healthy range
    }

    risk_drivers = explanation[
        explanation["direction"] == "increases_risk"
    ].copy()

    reasons = []
    for _, row in risk_drivers.iterrows():
        feat  = row["feature"]
        value = row.get("value", None)

        # Always suppress internal/interaction features
        if feat in ALWAYS_SUPPRESS:
            continue

        # Suppress if actual value is within normal/acceptable range
        if feat in SUPPRESS_IF_BELOW and value is not None:
            threshold = SUPPRESS_IF_BELOW[feat]
            try:
                if float(value) < threshold:
                    continue
            except (TypeError, ValueError):
                pass

        # Never use raw feature name in a consumer-facing reason
        reason = REASON_MAP.get(feat)
        if reason is None:
            # Skip unmapped features rather than expose raw names
            continue

        if reason not in reasons:  # Deduplicate
            reasons.append(reason)

        if len(reasons) >= top_n:
            break

    return reasons
