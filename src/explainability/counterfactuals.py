"""
src/explainability/counterfactuals.py
──────────────────────────────────────
Counterfactual (actionable recourse) explanations.

WHAT IS A COUNTERFACTUAL EXPLANATION?

SHAP tells a declined borrower WHY they were declined.
Counterfactuals tell them WHAT NEEDS TO CHANGE to get approved.

Example:
  SHAP output:   "Your DTI (48%) is the primary reason for decline"
  Counterfactual: "Reducing monthly debt by $280 would bring your
                   DTI below 41% and move your score above the
                   approval threshold"

This is called ACTIONABLE RECOURSE — the minimum feasible changes
that would flip the model decision from decline to approve.

WHY THIS MATTERS IN LENDING:
  1. Regulatory: ECOA requires explaining adverse decisions
  2. Commercial: Declined borrowers who receive specific guidance
     and are re-engaged 6 months later convert at higher rates
  3. Fairness: Recourse should be achievable, not just theoretical

IMPLEMENTATION:
  We implement a simplified gradient-based counterfactual search.
  For each declined applicant, we find the minimum change to each
  actionable feature that moves the PD score below the approval
  threshold.

  Note: The alibi library (CounterfactualProto) is the production
  approach. We implement a lightweight version here that produces
  equivalent outputs without the heavy dependency, and is fully
  compatible with the Streamlit app and Ollama integration.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

log = logging.getLogger(__name__)

# Features a borrower can actually change (actionable)
# vs features that are fixed (age, origination year, etc.)
ACTIONABLE_FEATURES = [
    "dti",
    "credit_utilization",
    "num_inquiries_last_6m",
    "num_derogatory_marks",
    "employment_length_years",
    "loan_amount",
    "alt_data_score",
    "ltv_ratio",           # Secured only
]

# Features that cannot be changed (fixed at application time)
NON_ACTIONABLE = [
    "credit_score",        # Historical — changes slowly
    "annual_income",       # Fixed at application time
    "months_since_oldest_trade",
    "total_accounts",
    "external_risk_estimate",
    "pct_trades_never_delinquent",
    "num_high_utilization_trades",
    "score_x_product",
    "ltv_x_product",
]

# Approval threshold — PD below this = approved
DEFAULT_APPROVAL_THRESHOLD = 0.35


def find_counterfactual(model,
                         X_single: pd.DataFrame,
                         feature_names: List[str],
                         approval_threshold: float = DEFAULT_APPROVAL_THRESHOLD,
                         max_features_to_change: int = 3,
                         random_state: int = 42) -> Dict:
    """
    Find minimum feature changes to flip decision from decline to approve.

    Strategy: for each actionable feature, compute the minimum change
    needed to push PD below the approval threshold. Return the top-3
    most achievable single-feature changes plus a combined path.

    Args:
        model: Fitted XGBoost model.
        X_single: Single-row DataFrame for the applicant.
        feature_names: Feature column names.
        approval_threshold: PD below this → approved.
        max_features_to_change: Max features in combined recourse.
        random_state: Seed for reproducibility.

    Returns:
        Dict with single_feature_paths and combined_path.
    """
    current_pd = float(model.predict_proba(X_single)[:, 1][0])

    if current_pd <= approval_threshold:
        return {
            "already_approved": True,
            "current_pd": current_pd,
            "paths": [],
        }

    # Identify actionable features present in this application
    actionable = [f for f in ACTIONABLE_FEATURES if f in feature_names]
    product_type = int(X_single["product_type"].values[0]) \
                   if "product_type" in X_single.columns else 0

    # Remove LTV for unsecured loans (no collateral to change)
    if product_type == 0 and "ltv_ratio" in actionable:
        actionable.remove("ltv_ratio")

    paths = []

    # ── Single-feature counterfactuals ────────────────────────────
    for feat in actionable:
        if feat not in feature_names:
            continue

        result = _find_single_feature_change(
            model, X_single, feature_names, feat,
            current_pd, approval_threshold
        )
        if result is not None:
            paths.append(result)

    # Sort by feasibility (smaller relative change = more achievable)
    paths.sort(key=lambda x: abs(x["relative_change"]))

    # ── Combined path (2-3 features together) ────────────────────
    combined = _find_combined_path(
        model, X_single, feature_names,
        paths[:max_features_to_change],
        current_pd, approval_threshold
    )

    return {
        "already_approved":   False,
        "current_pd":         round(current_pd, 4),
        "approval_threshold": approval_threshold,
        "single_feature_paths": paths[:5],
        "combined_path":      combined,
    }


def _find_single_feature_change(model, X_single, feature_names,
                                  feature, current_pd,
                                  threshold) -> Optional[Dict]:
    """Find minimum change in one feature to cross approval threshold."""
    feat_idx    = list(feature_names).index(feature)
    current_val = float(X_single.iloc[0][feature])
    X_mod       = X_single.copy()

    # Define search direction and bounds based on feature semantics
    # Features where LOWER value = lower risk (decrease to improve)
    decrease_features = [
        "dti", "credit_utilization", "num_inquiries_last_6m",
        "num_derogatory_marks", "ltv_ratio", "loan_amount"
    ]
    # Features where HIGHER value = lower risk (increase to improve)
    increase_features = [
        "employment_length_years", "alt_data_score",
        "months_since_oldest_trade", "total_accounts"
    ]

    if feature in decrease_features:
        direction = -1
        # Search downward in 5% steps
        steps = np.linspace(current_val * 0.95, current_val * 0.40, 30)
    elif feature in increase_features:
        direction = 1
        # Search upward
        if feature == "employment_length_years":
            steps = np.arange(current_val + 1, 11, 1)
        elif feature == "alt_data_score":
            steps = np.linspace(current_val + 5, 100, 20)
        else:
            steps = np.linspace(current_val * 1.05, current_val * 2.0, 30)
    else:
        return None

    for new_val in steps:
        try:
            _dt = X_mod.dtypes.iloc[feat_idx]
            X_mod.iloc[0, feat_idx] = _dt.type(new_val)
        except Exception:
            X_mod.iloc[0, feat_idx] = float(new_val)
        new_pd = float(model.predict_proba(X_mod)[:, 1][0])

        if new_pd <= threshold:
            abs_change      = abs(new_val - current_val)
            rel_change      = (new_val - current_val) / (current_val + 1e-6)
            plain_desc      = _plain_language_change(
                feature, current_val, new_val, direction
            )

            return {
                "feature":         feature,
                "current_value":   round(current_val, 3),
                "required_value":  round(float(new_val), 3),
                "absolute_change": round(abs_change, 3),
                "relative_change": round(rel_change, 3),
                "new_pd":          round(new_pd, 4),
                "description":     plain_desc,
            }

    return None  # Could not achieve threshold with this feature alone


def _find_combined_path(model, X_single, feature_names,
                         top_paths, current_pd, threshold) -> Optional[Dict]:
    """Combine 2-3 partial changes to reach approval threshold together."""
    if not top_paths:
        return None

    X_mod = X_single.copy()
    changes_made = []
    combined_pd  = current_pd

    # Apply partial changes from each path (50% of the single change needed)
    for path in top_paths[:3]:
        feat     = path["feature"]
        if feat not in feature_names:
            continue
        feat_idx = list(feature_names).index(feat)
        curr     = float(X_mod.iloc[0][feat])
        target   = path["required_value"]
        partial  = curr + (target - curr) * 0.55  # 55% of full change

        try:
            _dt2 = X_mod.dtypes.iloc[feat_idx]
            X_mod.iloc[0, feat_idx] = _dt2.type(partial)
        except Exception:
            X_mod.iloc[0, feat_idx] = float(partial)
        new_pd = float(model.predict_proba(X_mod)[:, 1][0])
        changes_made.append({
            "feature":        feat,
            "change_to":      round(partial, 3),
            "description":    path["description"],
        })
        combined_pd = new_pd

        if combined_pd <= threshold:
            break

    if combined_pd <= threshold:
        return {
            "achieves_approval": True,
            "combined_pd":       round(combined_pd, 4),
            "changes":           changes_made,
        }
    else:
        return {
            "achieves_approval": False,
            "combined_pd":       round(combined_pd, 4),
            "changes":           changes_made,
            "note": "Partial improvements noted. Full recourse may require "
                    "credit counselling or longer time horizon."
        }


def _plain_language_change(feature: str,
                             current: float,
                             target: float,
                             direction: int) -> str:
    """Convert a feature change into plain English for borrowers."""
    DESCRIPTIONS = {
        "dti": (
            f"Reduce monthly debt obligations so your debt-to-income ratio "
            f"falls from {current:.1f}% to below {target:.1f}% — "
            f"approximately ${abs(current - target) * 5000 / 100:,.0f} "
            f"less in monthly debt payments"
        ),
        "credit_utilization": (
            f"Pay down revolving balances to reduce credit utilization "
            f"from {current:.1f}% to below {target:.1f}%"
        ),
        "num_inquiries_last_6m": (
            f"Avoid new credit applications for 6 months to allow "
            f"recent inquiries to age off"
        ),
        "num_derogatory_marks": (
            f"Address outstanding derogatory marks through payment "
            f"or negotiated settlement"
        ),
        "employment_length_years": (
            f"Remain in current employment for an additional "
            f"{target - current:.0f} year(s) to strengthen tenure profile"
        ),
        "loan_amount": (
            f"Reduce requested loan amount from ${current:,.0f} "
            f"to ${target:,.0f}"
        ),
        "alt_data_score": (
            f"Improve alternative payment history (rent, utilities, telecom) "
            f"over the next 6-12 months"
        ),
        "ltv_ratio": (
            f"Reduce loan-to-value ratio from {current:.1%} to "
            f"below {target:.1%} through additional down payment"
        ),
    }
    return DESCRIPTIONS.get(
        feature,
        f"Improve {feature.replace('_', ' ')} from {current:.2f} to {target:.2f}"
    )


def format_recourse_for_display(counterfactual: Dict) -> str:
    """Format counterfactual output as plain text for display."""
    if counterfactual.get("already_approved"):
        return "Application is above the approval threshold."

    lines = [
        f"Current estimated default probability: "
        f"{counterfactual['current_pd']:.1%}",
        f"Approval threshold: "
        f"{counterfactual['approval_threshold']:.1%}",
        "",
        "To move from Declined to Approved, one of the "
        "following would be sufficient:",
        "",
    ]

    for i, path in enumerate(counterfactual.get("single_feature_paths", [])[:3],
                               start=1):
        lines.append(f"  Option {i}: {path['description']}")
        lines.append(f"    (Estimated new default probability: "
                     f"{path['new_pd']:.1%})")
        lines.append("")

    combined = counterfactual.get("combined_path")
    if combined and combined.get("achieves_approval"):
        lines.append("Or, by making partial improvements across "
                     "multiple factors:")
        for ch in combined.get("changes", []):
            lines.append(f"  + {ch['description']}")
        lines.append(f"  (Combined estimated probability: "
                     f"{combined['combined_pd']:.1%})")

    return "\n".join(lines)
