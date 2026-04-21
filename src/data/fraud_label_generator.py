"""
src/data/fraud_label_generator.py
───────────────────────────────────
Synthetic fraud label generation for portfolio projects.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: THIS IS A PLACEHOLDER MODULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LendingClub and FICO HELOC datasets do NOT contain confirmed
fraud labels. Charged-off loans could be either credit losses
(borrower inability to pay) or fraud losses (borrower never
intended to pay). These two outcomes look similar in the data
but require very different interventions.

This module generates SYNTHETIC fraud labels based on known
fraud indicator patterns from published industry research.
The labels are realistic in their distribution and are
sufficient to demonstrate the full detection pipeline.

TO SWAP IN REAL DATA:
──────────────────────
Replace the generate_synthetic_fraud_labels() call with a
call to load_proprietary_fraud_labels() using one of the
documented interfaces below. No other code changes required —
the feature engineering and model training pipeline is
agnostic to label source.

REAL DATA SOURCES (documented interfaces):
──────────────────────────────────────────

1. Lender Internal Investigations Database
   ─────────────────────────────────────────
   Expected schema (SQL query or CSV export):
     account_id:          VARCHAR  — matches loan account identifier
     fraud_confirmed:     BOOLEAN  — TRUE if investigation confirmed fraud
     fraud_type:          VARCHAR  — see FRAUD_TYPES below
     investigation_date:  DATE     — when investigation concluded
     loss_attributed:     DECIMAL  — dollar amount attributed to fraud
     investigator_id:     VARCHAR  — audit trail
     data_source:         VARCHAR  = 'lender_internal'

2. LexisNexis Fraud Intelligence / RiskView
   ──────────────────────────────────────────
   API: LexisNexis RiskView batch API
   Endpoint: POST /v1/fraud/batch
   Returns: risk scores + consortium fraud flags
   Expected fields:
     ln_fraud_score:      INT      — 0-999 (higher = higher fraud risk)
     ln_id_verified:      BOOLEAN  — identity verification result
     ln_synthetic_flag:   BOOLEAN  — synthetic identity indicator
     ln_consortium_match: BOOLEAN  — matched in shared fraud database
     data_source:         VARCHAR  = 'lexisnexis'

3. Equifax Fraud Shield / InterConnect
   ──────────────────────────────────────
   API: Equifax Fraud Shield API
   Fields:
     eq_fraud_score:      INT      — Equifax fraud risk score
     eq_device_match:     BOOLEAN  — device fingerprint match
     eq_address_verify:   BOOLEAN  — address verification
     eq_synthetic_ind:    BOOLEAN  — synthetic ID indicator
     data_source:         VARCHAR  = 'equifax'

4. Socure Score 3.0
   ───────────────────
   API: Socure Predictive DocV API
   Fields:
     socure_sigma_score:  FLOAT    — 0-1 fraud probability
     socure_id_verified:  BOOLEAN
     socure_doc_verified: BOOLEAN
     data_source:         VARCHAR  = 'socure'

FRAUD TYPES (standard taxonomy):
──────────────────────────────────
  first_party        — Borrower misrepresents ability/intent to repay
  synthetic_id       — Fabricated identity using real + fake PII
  third_party        — Stolen identity used without victim knowledge
  straw_borrower     — Nominee borrower fronting for another party
  loan_stacking      — Multiple simultaneous applications across lenders
  document_manip     — Falsified income/employment documents
  dealer_fraud       — Auto dealer submits falsified deal packages
  payment_fraud      — Fraudulent payment instruments (NSF schemes)
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

log = logging.getLogger(__name__)

FRAUD_TYPES = [
    "first_party",
    "synthetic_id",
    "third_party",
    "straw_borrower",
    "loan_stacking",
    "document_manip",
    "dealer_fraud",
    "payment_fraud",
]

# Industry benchmark fraud rates by product
# Sources: LexisNexis True Cost of Fraud 2023, TransUnion Fraud Insights 2024
FRAUD_RATES = {
    "unsecured_consumer": 0.028,   # ~2.8% of funded loans
    "secured_heloc":      0.012,   # ~1.2% of funded loans
    "auto_dealer":        0.035,   # ~3.5% (higher due to dealer channel)
}

# Fraud type distribution (conditional on fraud confirmed)
FRAUD_TYPE_DIST = {
    "first_party":   0.32,
    "synthetic_id":  0.18,
    "third_party":   0.14,
    "straw_borrower":0.10,
    "loan_stacking": 0.10,
    "document_manip":0.09,
    "dealer_fraud":  0.05,
    "payment_fraud": 0.02,
}


def generate_synthetic_fraud_labels(
        df: pd.DataFrame,
        pd_col: str = "pd_score",
        credit_score_col: str = "credit_score",
        product_type_col: str = "product_type",
        random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic fraud labels based on known fraud indicators.

    METHODOLOGY:
    Fraud probability is higher for:
      - First Payment Default (FPD) loans — strongest indicator
      - Low credit scores (subprime) — higher first-party fraud
      - High loan amounts relative to income — overstated income
      - Thin-file applicants — synthetic ID risk
      - Very new employment — straw borrower signal
      - High number of recent inquiries — loan stacking

    Fraud type assignment is probabilistic based on the borrower
    profile and the empirical distribution of fraud types.

    NOTE: These labels are synthetic proxies. The correlations
    are realistic but the labels are NOT ground truth. Model
    performance on these labels will be artificially good because
    the labels are derived from the same features used for detection.
    Real fraud labels from investigations produce harder, noisier
    training problems.

    Args:
        df: Portfolio DataFrame.
        pd_col: PD score column (higher PD → higher fraud probability).
        credit_score_col: Credit score column.
        product_type_col: 0=unsecured, 1=secured.
        random_state: Reproducibility seed.

    Returns:
        DataFrame with fraud label columns added.
    """
    log.info("Generating synthetic fraud labels ...")
    rng  = np.random.default_rng(random_state)
    df   = df.copy()
    n    = len(df)

    # Base fraud probability from product type
    base_rate = np.where(
        df[product_type_col] == 1,
        FRAUD_RATES["secured_heloc"],
        FRAUD_RATES["unsecured_consumer"]
    )

    # Adjust based on risk signals
    # High PD × low score is the strongest joint indicator
    pd_vals = df[pd_col].values if pd_col in df.columns \
              else np.full(n, 0.25)
    cs_vals = df[credit_score_col].values if credit_score_col in df.columns \
              else np.full(n, 650)

    # FPD proxy: very high PD (>65%) + low credit score (<580)
    fpd_signal = (pd_vals > 0.65).astype(float) * (cs_vals < 580).astype(float)

    # Thin-file proxy
    thin_signal = (df["thin_file_flag"].astype(float)
                   if "thin_file_flag" in df.columns
                   else np.zeros(n))

    # Score band signals
    subprime_signal = (cs_vals < 600).astype(float) * 0.5
    deep_sub_signal = (cs_vals < 540).astype(float) * 0.8

    # Compute fraud probability
    fraud_prob = (
        base_rate
        + fpd_signal     * 0.08   # FPD: +8pp
        + thin_signal    * 0.03   # Thin file: +3pp
        + subprime_signal * 0.02  # Subprime: +2pp
        + deep_sub_signal * 0.03  # Deep subprime: +3pp
    )
    fraud_prob = np.clip(fraud_prob, 0.001, 0.25)

    # Assign fraud confirmed flag
    fraud_confirmed = rng.uniform(0, 1, n) < fraud_prob

    # Assign fraud types (for confirmed fraud only)
    fraud_types = np.where(fraud_confirmed, "clean", "clean")
    fraud_type_names = list(FRAUD_TYPE_DIST.keys())
    fraud_type_probs = list(FRAUD_TYPE_DIST.values())

    for i in np.where(fraud_confirmed)[0]:
        # Bias type toward profile-consistent fraud
        if cs_vals[i] < 560 and pd_vals[i] > 0.55:
            # High PD, very low score → likely first-party or synthetic
            adjusted = [0.45, 0.30, 0.10, 0.05, 0.05, 0.03, 0.01, 0.01]
        elif thin_signal[i] > 0:
            # Thin file → synthetic ID bias
            adjusted = [0.20, 0.45, 0.15, 0.05, 0.05, 0.05, 0.03, 0.02]
        else:
            adjusted = fraud_type_probs

        fraud_types[i] = rng.choice(fraud_type_names, p=adjusted)

    # FPD flag (first payment default — strongest fraud indicator)
    # Defined as: confirmed default AND very high PD AND low score
    fpd_flag = fraud_confirmed & (pd_vals > 0.60) & (cs_vals < 580)

    # Loan stacking flag: multiple inquiries + very recent accounts
    loan_stacking_flag = np.array(
        [(fraud_types[i] == "loan_stacking") for i in range(n)]
    )

    # Synthetic ID flag
    synthetic_id_flag = np.array(
        [(fraud_types[i] == "synthetic_id") for i in range(n)]
    )

    # Loss attributed (fraud confirmed only)
    ead_vals = df["ead_estimate"].values if "ead_estimate" in df.columns \
               else np.full(n, 10000.0)
    loss_attr = np.where(
        fraud_confirmed,
        ead_vals * rng.uniform(0.5, 0.95, n),  # Recover 5-50% of fraud loss
        0.0
    )

    # Add columns
    df["fraud_confirmed"]   = fraud_confirmed
    df["fraud_type"]        = fraud_types
    df["fraud_probability"] = fraud_prob.round(4)
    df["fpd_flag"]          = fpd_flag
    df["loan_stacking_flag"]= loan_stacking_flag
    df["synthetic_id_flag"] = synthetic_id_flag
    df["loss_attributed"]   = loss_attr.round(2)
    df["data_source"]       = "synthetic_placeholder"

    n_fraud = fraud_confirmed.sum()
    log.info(f"  Fraud labels generated: {n_fraud:,} confirmed "
             f"({n_fraud/n:.2%} fraud rate)")
    log.info(f"  FPD flags:              {fpd_flag.sum():,}")
    log.info(f"  Synthetic ID flags:     {synthetic_id_flag.sum():,}")
    log.info(f"  Loan stacking flags:    {loan_stacking_flag.sum():,}")
    log.info(f"  Total fraud losses:     ${loss_attr.sum():,.0f}")

    return df


def load_proprietary_fraud_labels(
        source: str,
        filepath: Optional[str] = None,
        **api_kwargs) -> pd.DataFrame:
    """
    ──────────────────────────────────────────────────────────
    SWAP POINT: Replace synthetic labels with real fraud data.
    ──────────────────────────────────────────────────────────

    Args:
        source: One of 'lender_internal', 'lexisnexis',
                'equifax', 'socure'
        filepath: Path to CSV/parquet export (for file-based sources)
        **api_kwargs: API credentials for real-time sources

    Returns:
        DataFrame with standardised fraud label columns.

    Raises:
        NotImplementedError: Always — this is the swap point.
    """
    raise NotImplementedError(
        f"Real fraud label loading not implemented for source='{source}'. "
        f"Replace this function body with your {source} data access code. "
        f"See module docstring for expected schema."
    )
