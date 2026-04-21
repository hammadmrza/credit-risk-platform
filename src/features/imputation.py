"""
src/features/imputation.py
──────────────────────────
Missing value imputation for the credit risk platform.

Strategy decisions (from Phase 2 EDA findings):

1. months_since_recent_delinquency
   NaN = borrower has NEVER been delinquent → excellent signal
   Strategy: fill with 999 (sentinel = "very long time ago / never")
   Rationale: preserves the signal that no-delinquency-ever is positive

2. ltv_ratio (unsecured loans, product_type = 0)
   NaN = no collateral (correct, expected for personal loans)
   Strategy: fill with 0.0
   Rationale: tree models treat 0 LTV correctly as "no collateral risk"

3. external_risk_estimate, pct_trades_never_delinquent,
   months_since_recent_trade, num_high_utilization_trades
   NaN = LendingClub rows (HELOC-only features)
   Strategy: fill with -1 sentinel
   Rationale: tree models learn -1 means "not applicable for this product"

4. loan_term_months, num_open_accounts
   NaN = HELOC rows (LC-only features)
   Strategy: fill with median of LC segment
   Rationale: HELOC rows need a plausible value; median is conservative

5. All remaining numeric NaN (< 5% null rate)
   Strategy: median imputation by product_type segment
   Rationale: segment-aware imputation preserves product distributions
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

log = logging.getLogger(__name__)

SENTINEL_NEVER_DELINQUENT  = 999.0
SENTINEL_NOT_APPLICABLE    = -1.0
SENTINEL_LTV_UNSECURED     = 0.0


def impute(df: pd.DataFrame,
           medians: Optional[Dict] = None,
           fit: bool = True) -> tuple:
    """
    Apply full imputation strategy to the dataset.

    Call with fit=True on training data to compute medians.
    Call with fit=False + medians dict on test data to apply same medians.

    Args:
        df: DataFrame to impute.
        medians: Pre-computed median dict (for test set).
        fit: If True, compute medians from df. If False, use provided medians.

    Returns:
        (imputed_df, medians_dict) tuple.
    """
    df = df.copy()
    if medians is None:
        medians = {}

    # ── Rule 1: Never-delinquent sentinel ────────────────────────
    col = "months_since_recent_delinquency"
    if col in df.columns:
        n_filled = df[col].isna().sum()
        df[col] = df[col].fillna(SENTINEL_NEVER_DELINQUENT)
        log.info(f"  {col}: {n_filled:,} NaN → {SENTINEL_NEVER_DELINQUENT} "
                 f"(never delinquent sentinel)")

    # ── Rule 2: LTV = 0 for unsecured ────────────────────────────
    col = "ltv_ratio"
    if col in df.columns:
        unsec_null = (df["product_type"] == 0) & df[col].isna()
        df.loc[unsec_null, col] = SENTINEL_LTV_UNSECURED
        log.info(f"  ltv_ratio: {unsec_null.sum():,} unsecured NaN → 0.0")

    # ── Rule 3: HELOC-only features → -1 for LC rows ─────────────
    heloc_only = [
        "external_risk_estimate",
        "pct_trades_never_delinquent",
        "months_since_recent_trade",
        "num_high_utilization_trades",
    ]
    for col in heloc_only:
        if col in df.columns:
            n = df[col].isna().sum()
            df[col] = df[col].fillna(SENTINEL_NOT_APPLICABLE)
            log.info(f"  {col}: {n:,} NaN → -1 (not applicable sentinel)")

    # ── Rule 4: LC-only features → segment median for HELOC ──────
    lc_only_numeric = ["loan_term_months", "num_open_accounts"]
    for col in lc_only_numeric:
        if col not in df.columns:
            continue
        if fit:
            median_val = df[df["product_type"] == 0][col].median()
            medians[col] = median_val
        else:
            median_val = medians.get(col, df[col].median())
        n = df[col].isna().sum()
        df[col] = df[col].fillna(median_val)
        log.info(f"  {col}: {n:,} NaN → {median_val:.1f} (LC segment median)")

    # ── Rule 5: All remaining numeric NaN → segment median ───────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    remaining_null_cols = [c for c in numeric_cols
                           if df[c].isna().sum() > 0
                           and c not in ["default_flag",
                                         "origination_year",
                                         "product_type"]]

    for col in remaining_null_cols:
        if fit:
            seg_medians = df.groupby("product_type")[col].median().to_dict()
            medians[f"{col}_by_segment"] = seg_medians
        else:
            seg_medians = medians.get(f"{col}_by_segment", {})

        for seg in df["product_type"].unique():
            mask = (df["product_type"] == seg) & df[col].isna()
            if mask.sum() > 0:
                fill_val = seg_medians.get(
                    seg,
                    df[df["product_type"] == seg][col].median()
                )
                df.loc[mask, col] = fill_val
                log.info(f"  {col} (seg={seg}): {mask.sum():,} NaN → "
                         f"{fill_val:.2f} (segment median)")

    # Verify no numeric nulls remain (except known allowed ones)
    allowed_null = ["origination_date", "has_synthetic_features"]
    remaining = df.select_dtypes(include=[np.number]).isnull().sum()
    remaining = remaining[remaining > 0]
    if len(remaining) > 0:
        log.warning(f"Remaining nulls after imputation: {remaining.to_dict()}")
    else:
        log.info("  All numeric nulls resolved.")

    return df, medians
