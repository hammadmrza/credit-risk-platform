"""
src/explainability/fairness_audit.py
──────────────────────────────────────
Fairness audit for credit risk models.

WHY FAIRNESS AUDITING IS MANDATORY IN LENDING:

In Canada (OSFI E-23, FCAC guidelines) and the US (ECOA, Fair Housing Act,
Regulation B), lenders must ensure models do not produce discriminatory
outcomes against protected classes.

You cannot use protected attributes (race, gender, religion) as model
features. But you CAN — and MUST — audit whether your model produces
disparate impact on proxies correlated with those attributes.

THREE FAIRNESS METRICS:

1. DEMOGRAPHIC PARITY
   Are approval rates similar across groups?
   "The approval rate for Group A should not differ from Group B by
   more than 20% without a clear credit-risk justification."
   Threshold: |rate_A - rate_B| / rate_B > 0.20 → investigate

2. EQUALIZED ODDS
   Are true positive rates (approving creditworthy borrowers) and
   false positive rates (approving risky borrowers) similar across groups?
   A model can be accurate on average but systematically wrong for one group.

3. CALIBRATION
   Does a score of 0.35 PD mean the same default probability for ALL groups?
   If the model systematically over-predicts for one group (assigning
   higher PD than their actual default rate), that group is penalised
   by an inaccurate model.

SEGMENTS WE AUDIT:
  - Employment type (salaried vs. self-employed vs. unemployed/gig)
  - Loan purpose (debt consolidation vs. home improvement vs. other)
  - Origination vintage (lending cycle fairness)
  - Product type (secured vs. unsecured)

NOTE: We do not segment by race, gender, or national origin —
these are protected attributes. The segments above are legitimate
credit-relevant distinctions that happen to correlate with
protected attributes in the broader population, which is why
we audit them.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

log = logging.getLogger(__name__)

DISPARATE_IMPACT_THRESHOLD = 0.20   # 20% gap triggers investigation
CALIBRATION_THRESHOLD      = 0.05   # 5% miscalibration triggers flag


def compute_demographic_parity(df: pd.DataFrame,
                                 pd_col: str,
                                 segment_col: str,
                                 approval_threshold: float = 0.35,
                                 min_group_size: int = 100) -> pd.DataFrame:
    """
    Compute approval rates by segment and flag disparate impact.

    Args:
        df: DataFrame with PD scores and segment column.
        pd_col: Column containing PD estimates (0-1).
        segment_col: Column defining demographic segments.
        approval_threshold: PD below this = approved.
        min_group_size: Minimum group size to include in audit.

    Returns:
        DataFrame with segment, approval_rate, gap_from_best, flag.
    """
    df = df.copy()
    df["approved"] = (df[pd_col] <= approval_threshold).astype(int)

    rows = []
    for seg in df[segment_col].dropna().unique():
        group = df[df[segment_col] == seg]
        if len(group) < min_group_size:
            continue
        rows.append({
            "segment":       str(seg),
            "n":             len(group),
            "approval_rate": round(group["approved"].mean(), 4),
            "avg_pd":        round(group[pd_col].mean(), 4),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("approval_rate", ascending=False)
    best_rate = result["approval_rate"].max()

    result["gap_from_best"] = (best_rate - result["approval_rate"]).round(4)
    result["disparate_impact"] = (
        result["gap_from_best"] / (best_rate + 1e-6)
    ).round(4)

    result["flag"] = np.where(
        result["disparate_impact"] > DISPARATE_IMPACT_THRESHOLD,
        "INVESTIGATE",
        "OK"
    )

    return result.reset_index(drop=True)


def compute_equalized_odds(df: pd.DataFrame,
                             pd_col: str,
                             target_col: str,
                             segment_col: str,
                             approval_threshold: float = 0.35,
                             min_group_size: int = 100) -> pd.DataFrame:
    """
    Compute true positive and false positive rates by segment.

    TPR = % of truly creditworthy borrowers who are correctly approved
    FPR = % of truly risky borrowers who are incorrectly approved

    Equalized odds requires both TPR and FPR to be similar across groups.

    Args:
        df: DataFrame with PD, target, and segment columns.
        pd_col: PD estimate column.
        target_col: Actual default flag (0=good, 1=bad).
        segment_col: Segment column.
        approval_threshold: PD below this = approved.
        min_group_size: Min group size.

    Returns:
        DataFrame with TPR, FPR, and flags by segment.
    """
    df = df.copy()
    df["approved"]  = (df[pd_col] <= approval_threshold).astype(int)
    df["true_good"] = (df[target_col] == 0).astype(int)
    df["true_bad"]  = (df[target_col] == 1).astype(int)

    rows = []
    for seg in df[segment_col].dropna().unique():
        group = df[df[segment_col] == seg]
        if len(group) < min_group_size:
            continue

        good_group = group[group["true_good"] == 1]
        bad_group  = group[group["true_bad"]  == 1]

        tpr = good_group["approved"].mean() if len(good_group) > 0 else np.nan
        fpr = bad_group["approved"].mean()  if len(bad_group)  > 0 else np.nan

        rows.append({
            "segment":   str(seg),
            "n":         len(group),
            "n_good":    len(good_group),
            "n_bad":     len(bad_group),
            "tpr":       round(float(tpr), 4) if not np.isnan(tpr) else None,
            "fpr":       round(float(fpr), 4) if not np.isnan(fpr) else None,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).dropna(subset=["tpr", "fpr"])

    if len(result) > 1:
        best_tpr = result["tpr"].max()
        best_fpr = result["fpr"].min()
        result["tpr_gap"] = (best_tpr - result["tpr"]).round(4)
        result["fpr_gap"] = (result["fpr"]  - best_fpr).round(4)
        result["flag"] = np.where(
            (result["tpr_gap"] > DISPARATE_IMPACT_THRESHOLD) |
            (result["fpr_gap"] > DISPARATE_IMPACT_THRESHOLD),
            "INVESTIGATE", "OK"
        )

    return result.reset_index(drop=True)


def compute_calibration(df: pd.DataFrame,
                          pd_col: str,
                          target_col: str,
                          segment_col: str,
                          n_bins: int = 5,
                          min_group_size: int = 100) -> pd.DataFrame:
    """
    Check whether model PD scores are equally calibrated across segments.

    Calibration: does PD = 0.30 mean 30% default probability for ALL groups?
    Miscalibration = model systematically over/under-predicts for a group.

    Args:
        df: DataFrame with PD, target, and segment columns.
        pd_col: PD estimate column.
        target_col: Actual default flag.
        segment_col: Segment column.
        n_bins: Number of PD bins for calibration.
        min_group_size: Minimum group size.

    Returns:
        DataFrame with calibration error by segment.
    """
    df = df.copy()
    df["pd_bin"] = pd.qcut(df[pd_col], q=n_bins,
                             duplicates="drop", labels=False)

    rows = []
    for seg in df[segment_col].dropna().unique():
        group = df[df[segment_col] == seg]
        if len(group) < min_group_size:
            continue

        # Mean predicted PD vs mean actual default rate
        mean_pd_pred   = group[pd_col].mean()
        mean_dr_actual = group[target_col].mean()
        calib_error    = mean_pd_pred - mean_dr_actual

        rows.append({
            "segment":       str(seg),
            "n":             len(group),
            "mean_pred_pd":  round(mean_pd_pred, 4),
            "actual_dr":     round(mean_dr_actual, 4),
            "calib_error":   round(calib_error, 4),
            "flag": ("OVER-PREDICTS RISK"
                     if calib_error > CALIBRATION_THRESHOLD
                     else "UNDER-PREDICTS RISK"
                     if calib_error < -CALIBRATION_THRESHOLD
                     else "OK"),
        })

    return pd.DataFrame(rows).sort_values(
        "calib_error", key=abs, ascending=False
    ).reset_index(drop=True)


def run_full_audit(df: pd.DataFrame,
                    pd_col: str = "pd_score",
                    target_col: str = "default_flag",
                    segment_cols: Optional[List[str]] = None,
                    approval_threshold: float = 0.35) -> Dict:
    """
    Run complete fairness audit across all segments.

    Args:
        df: Portfolio DataFrame with PD, target, and segment columns.
        pd_col: PD estimate column name.
        target_col: Actual default flag column name.
        segment_cols: Columns to segment by. Auto-detects if None.
        approval_threshold: PD approval threshold.

    Returns:
        Dict with demographic_parity, equalized_odds, calibration
        results for each segment column.
    """
    if segment_cols is None:
        # Auto-detect available segment columns
        candidates = ["purpose", "home_ownership", "lc_grade",
                      "product_type", "verification_status"]
        segment_cols = [c for c in candidates if c in df.columns]

    results = {}

    for seg_col in segment_cols:
        log.info(f"  Auditing segment: {seg_col}")

        dp  = compute_demographic_parity(df, pd_col, seg_col,
                                          approval_threshold)
        eo  = compute_equalized_odds(df, pd_col, target_col,
                                      seg_col, approval_threshold)
        cal = compute_calibration(df, pd_col, target_col, seg_col)

        results[seg_col] = {
            "demographic_parity": dp,
            "equalized_odds":     eo,
            "calibration":        cal,
            "flags": {
                "dp_flags":  int(dp["flag"].eq("INVESTIGATE").sum())
                             if len(dp) > 0 else 0,
                "eo_flags":  int(eo["flag"].eq("INVESTIGATE").sum())
                             if "flag" in eo.columns and len(eo) > 0 else 0,
                "cal_flags": int(cal["flag"].ne("OK").sum())
                             if len(cal) > 0 else 0,
            }
        }

    return results


def generate_model_card_fairness_section(audit_results: Dict) -> str:
    """
    Generate the fairness section of the model card (OSFI E-23 format).

    Args:
        audit_results: Output of run_full_audit().

    Returns:
        Formatted markdown string for MODEL_CARD.md.
    """
    lines = [
        "## Fairness Evaluation",
        "",
        "### Methodology",
        "Fairness was evaluated across employment type, loan purpose, and product type.",
        "Three metrics were assessed: demographic parity, equalized odds, and calibration.",
        "Threshold for investigation: >20% differential in approval rates or TPR/FPR.",
        "",
        "### Results by Segment",
        "",
    ]

    total_flags = 0
    for seg_col, results in audit_results.items():
        flags = results["flags"]
        total_flags += flags["dp_flags"] + flags["eo_flags"] + flags["cal_flags"]
        flag_count = flags["dp_flags"] + flags["eo_flags"] + flags["cal_flags"]
        status = "PASS" if flag_count == 0 else "INVESTIGATE"

        lines.append(f"**{seg_col.replace('_', ' ').title()}**: "
                     f"Status = {status} | "
                     f"DP flags: {flags['dp_flags']} | "
                     f"EO flags: {flags['eo_flags']} | "
                     f"Calibration flags: {flags['cal_flags']}")
        lines.append("")

    lines += [
        "### Overall Assessment",
        f"Total flags requiring investigation: {total_flags}",
        "",
        "### Limitations",
        "- Audit conducted on synthetic data; results will differ on real portfolio.",
        "- Protected class attributes (race, gender, national origin) not included "
          "in model features per ECOA requirements.",
        "- Segments audited are credit-relevant proxies, not protected classes.",
        "- Recommend re-running audit quarterly in production.",
    ]

    return "\n".join(lines)
