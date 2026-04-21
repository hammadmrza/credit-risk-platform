"""
src/features/fraud_features.py
────────────────────────────────
Fraud-specific feature engineering.

FEATURE CATEGORIES:
  1. Application-time red flags     — signals visible before funding
  2. Post-funding behavioural       — signals visible after funding
  3. Velocity and stacking          — cross-account patterns
  4. Dealer/channel anomalies       — channel concentration risk
  5. Identity consistency           — PII coherence signals

Each feature has a documented real-data source showing what
would power it in production (vs the synthetic proxy used here).
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

log = logging.getLogger(__name__)


def build_fraud_features(
        df: pd.DataFrame,
        pd_col: str = "pd_score",
        random_state: int = 42) -> pd.DataFrame:
    """
    Engineer fraud detection features from portfolio data.

    Args:
        df: DataFrame with credit model features + fraud labels.
        pd_col: PD score column from credit model.
        random_state: Reproducibility seed.

    Returns:
        DataFrame with fraud features added.
    """
    log.info("Engineering fraud detection features ...")
    rng = np.random.default_rng(random_state)
    df  = df.copy()
    n   = len(df)

    cs  = df["credit_score"].values \
          if "credit_score" in df.columns else np.full(n, 650.0)
    pd_v = df[pd_col].values \
           if pd_col in df.columns else np.full(n, 0.25)
    ead  = df["ead_estimate"].values \
           if "ead_estimate" in df.columns else np.full(n, 10000.0)

    # ── 1. APPLICATION-TIME FEATURES ─────────────────────────────

    # Income-to-loan ratio flag
    # REAL DATA: application form income / loan_amount
    # PROXY: estimate from EAD and credit tier
    if "annual_income" in df.columns and "loan_amount" in df.columns:
        df["income_loan_ratio"] = (
            df["annual_income"] / df["loan_amount"].clip(1)
        ).round(2)
        df["high_loan_to_income"] = (df["income_loan_ratio"] < 2.0).astype(int)
    else:
        df["income_loan_ratio"]    = (30000 / ead.clip(1)).round(2)
        df["high_loan_to_income"]  = (df["income_loan_ratio"] < 2.0).astype(int)

    # Employment tenure risk
    # REAL DATA: employment_length_years from application
    # PROXY: synthetic from feature config
    if "employment_length_years" in df.columns:
        df["short_tenure_flag"] = (
            df["employment_length_years"] < 1
        ).astype(int)
    else:
        df["short_tenure_flag"] = (
            rng.uniform(0, 1, n) < 0.12
        ).astype(int)

    # Score–income inconsistency
    # Borrowers with high scores but very low income relative
    # to loan amount may be misrepresenting income
    score_norm = (cs - 300) / (850 - 300)
    df["score_income_inconsistency"] = np.where(
        (score_norm > 0.6) & (df["income_loan_ratio"] < 1.5),
        1, 0
    )

    # ── 2. FIRST PAYMENT DEFAULT SIGNALS ─────────────────────────

    # FPD flag proxy
    # REAL DATA: payment_date of first scheduled payment vs due_date
    # PROXY: very high PD + subprime score = high FPD risk
    df["fpd_risk_score"] = (
        pd_v * (1 - score_norm)
    ).round(4)  # High PD × low score = highest FPD risk

    df["fpd_risk_flag"] = (
        (pd_v > 0.55) & (cs < 580)
    ).astype(int)

    # REAL DATA: days_past_due on first payment date
    df["epd_within_90d"] = (
        df["fpd_flag"] if "fpd_flag" in df.columns
        else (rng.uniform(0,1,n) < df["fpd_risk_score"] * 0.4).astype(int)
    )

    # ── 3. VELOCITY AND LOAN STACKING ────────────────────────────

    # Loan stacking score
    # REAL DATA: number of active applications within 30 days
    #            from bureau inquiry data or consortium database
    # PROXY: from synthetic label + inquiry signals
    if "num_inquiries_last_6m" in df.columns:
        inq = df["num_inquiries_last_6m"].values
    else:
        inq = rng.integers(0, 6, n).astype(float)

    df["velocity_score"] = (
        (inq / 6.0) * 0.6 +
        df.get("loan_stacking_flag", pd.Series(
            rng.uniform(0, 1, n) < 0.03, index=df.index
        )).astype(float) * 0.4
    ).round(4)

    df["high_velocity_flag"] = (df["velocity_score"] > 0.4).astype(int)

    # Multiple application flag
    # REAL DATA: dedup applicant_id or SSN across applications in window
    # PROXY: simulate 3% of portfolio with stacking patterns
    df["multi_app_flag"] = (
        df["loan_stacking_flag"] if "loan_stacking_flag" in df.columns
        else (rng.uniform(0, 1, n) < 0.03).astype(int)
    )

    # ── 4. SYNTHETIC IDENTITY SIGNALS ────────────────────────────

    # Synthetic ID score
    # REAL DATA: LexisNexis synthetic score, Socure Sigma,
    #            Equifax synthetic indicator
    # PROXY: thin-file + young account age = higher synthetic risk
    thin_flag = (df["thin_file_flag"].astype(float)
                 if "thin_file_flag" in df.columns
                 else (rng.uniform(0, 1, n) < 0.15).astype(float))

    months_oldest = (df["months_since_oldest_trade"].values
                     if "months_since_oldest_trade" in df.columns
                     else rng.uniform(6, 120, n))

    # Synthetic IDs tend to have short credit histories
    age_score = 1 - np.clip(np.array(months_oldest, dtype=float) / 120, 0, 1)

    df["synthetic_id_score"] = (
        thin_flag * 0.5 +
        age_score * 0.3 +
        (cs < 560).astype(float) * 0.2
    ).round(4)

    df["synthetic_id_risk_flag"] = (
        df["synthetic_id_score"] > 0.45
    ).astype(int)

    # ── 5. IDENTITY CONSISTENCY ───────────────────────────────────

    # Address stability flag
    # REAL DATA: bureau address history vs application address
    # PROXY: simulate 5% of portfolio with address inconsistency
    df["address_mismatch_flag"] = (
        rng.uniform(0, 1, n) < 0.05
    ).astype(int)

    # Document authenticity score
    # REAL DATA: document verification vendor (Jumio, Onfido, Persona)
    # PROXY: scaled fraud probability
    df["doc_verify_score"] = (
        1.0 - df.get("fraud_probability",
                      pd.Series(np.full(n, 0.03), index=df.index))
    ).clip(0, 1).round(3)

    # ── 6. COMPOSITE FRAUD SCORE ──────────────────────────────────

    df["fraud_feature_score"] = (
        df["fpd_risk_score"]       * 0.30 +
        df["velocity_score"]       * 0.20 +
        df["synthetic_id_score"]   * 0.20 +
        df["high_loan_to_income"]  * 0.10 +
        df["short_tenure_flag"]    * 0.08 +
        df["address_mismatch_flag"]* 0.07 +
        df["score_income_inconsistency"] * 0.05
    ).round(4)

    log.info(f"  Fraud features engineered: "
             f"{[c for c in df.columns if 'fraud' in c or 'fpd' in c or 'velocity' in c or 'synthetic' in c]}")

    return df


FRAUD_FEATURE_COLS = [
    "income_loan_ratio",
    "high_loan_to_income",
    "short_tenure_flag",
    "score_income_inconsistency",
    "fpd_risk_score",
    "fpd_risk_flag",
    "velocity_score",
    "high_velocity_flag",
    "multi_app_flag",
    "synthetic_id_score",
    "synthetic_id_risk_flag",
    "address_mismatch_flag",
    "doc_verify_score",
    "fraud_feature_score",
]
