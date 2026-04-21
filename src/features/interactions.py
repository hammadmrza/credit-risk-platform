"""
src/features/interactions.py
─────────────────────────────
Interaction feature engineering and reject inference.

INTERACTION FEATURES
────────────────────
Four interaction terms that capture how risk signals behave
differently depending on product type and borrower segment.

Each interaction is grounded in credit underwriting logic:

1. ltv_x_product_type
   LTV only matters for secured loans. For unsecured, LTV = 0
   (no collateral). The interaction fires for secured loans only.
   A HELOC with LTV = 0.85 is very different risk from LTV = 0.60.

2. dti_x_employment_length
   High DTI combined with short employment is a compound risk signal.
   A borrower with 48% DTI and 2 months tenure is far riskier than
   one with 48% DTI and 10 years tenure. The combination matters.

3. alt_data_x_thin_file
   ADS signal is most valuable for thin-file applicants (<3 tradelines).
   For standard-file borrowers, bureau data dominates. For thin-file,
   ADS carries proportionally more weight.

4. credit_score_x_product_type
   Score thresholds mean different things by product. A 620 score
   on a secured HELOC has collateral support; the same score on an
   unsecured personal loan does not. The interaction captures this.

REJECT INFERENCE
────────────────
Addresses sample selection bias: all training data comes from
approved-and-originated loans. Declined applicants are invisible.

Method: Augmentation
  Step 1: Train initial model on approved loans only
  Step 2: Generate a synthetic declined population
  Step 3: Score the declined population using the initial model
  Step 4: Assign probabilistic labels (high-risk declined → likely bad)
  Step 5: Combine original + weighted declined data
  Step 6: Retrain — this is the reject-inference-corrected dataset

Expected benefit: 2-4 AUC point improvement in the 550-620 score band
(where selection bias is strongest and financial inclusion decisions
are most consequential).
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

log = logging.getLogger(__name__)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add four interaction features to the dataset.

    Args:
        df: DataFrame with base features (post-imputation).

    Returns:
        DataFrame with four additional interaction columns.
    """
    df = df.copy()

    # ── 1. LTV × product_type ─────────────────────────────────────
    # Fires only for secured loans (product_type = 1)
    # For unsecured: ltv = 0.0, so interaction = 0.0
    if "ltv_ratio" in df.columns:
        df["ltv_x_product"] = (
            df["ltv_ratio"].fillna(0.0) * df["product_type"]
        )
        log.info("  ltv_x_product: LTV × secured flag")

    # ── 2. DTI × employment length ────────────────────────────────
    # High DTI + short tenure = compound risk signal
    # Normalised so both components are on similar scale
    if "dti" in df.columns and "employment_length_years" in df.columns:
        dti_norm  = df["dti"].clip(0, 80) / 80
        emp_norm  = df["employment_length_years"].clip(0, 10) / 10
        # Invert employment: short tenure = high risk contribution
        emp_risk  = 1 - emp_norm
        df["dti_x_emp_risk"] = (dti_norm * emp_risk).round(4)
        log.info("  dti_x_emp_risk: high DTI + short tenure compound signal")

    # ── 3. ADS × thin file ────────────────────────────────────────
    # ADS is most informative when bureau data is thin
    # For standard-file borrowers, this term contributes less
    if "alt_data_score" in df.columns and "thin_file_flag" in df.columns:
        thin_int = df["thin_file_flag"].astype(int)
        df["ads_x_thin_file"] = (
            df["alt_data_score"] * thin_int
        ).round(4)
        log.info("  ads_x_thin_file: ADS signal × thin-file flag")

    # ── 4. Credit score × product type ───────────────────────────
    # Captures different score thresholds by product
    if "credit_score" in df.columns:
        # Normalise score to 0-1
        score_norm = (df["credit_score"] - 300) / (850 - 300)
        # For secured: score has different implications (collateral present)
        df["score_x_product"] = (score_norm * df["product_type"]).round(4)
        log.info("  score_x_product: normalised score × secured flag")

    interactions_added = [
        c for c in ["ltv_x_product", "dti_x_emp_risk",
                     "ads_x_thin_file", "score_x_product"]
        if c in df.columns
    ]
    log.info(f"  Added {len(interactions_added)} interaction features: "
             f"{interactions_added}")
    return df


def generate_reject_inference(
        df_train: pd.DataFrame,
        n_declined: int = 5000,
        random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic declined population for reject inference.

    The declined population is parameterised to reflect the typical
    characteristics of rejected applicants:
      - Credit scores 80-120 points lower than approved population
      - DTI 10-15 points higher
      - More derogatory marks
      - Lower ADS (less alternative data support)
      - Implied high default rate (~55-65%)

    Args:
        df_train: Training DataFrame of approved loans.
        n_declined: Number of synthetic declined records to generate.
        random_state: Reproducibility seed.

    Returns:
        DataFrame of synthetic declined records with same schema.
    """
    log.info(f"Generating {n_declined:,} synthetic declined records ...")
    rng = np.random.default_rng(random_state)

    approved_score_mean = df_train["credit_score"].mean()
    approved_score_std  = df_train["credit_score"].std()
    approved_dti_mean   = df_train["dti"].mean()

    # Generate declined population — skewed toward poor credit quality
    credit_quality = rng.beta(2, 5, n_declined)  # Skewed left (poor quality)

    declined = pd.DataFrame({
        "credit_score": (
            approved_score_mean - 100 + credit_quality * 80
            + rng.normal(0, 25, n_declined)
        ).clip(300, 680),

        "dti": (
            approved_dti_mean + 15 - credit_quality * 20
            + rng.normal(0, 8, n_declined)
        ).clip(5, 80),

        "annual_income": np.exp(
            rng.normal(10.2, 0.6, n_declined)
        ).clip(15_000, 200_000),

        "credit_utilization": (
            70 - credit_quality * 45
            + rng.normal(0, 10, n_declined)
        ).clip(0, 100),

        "num_derogatory_marks": rng.choice(
            [0, 1, 2, 3, 4, 5], n_declined,
            p=[0.20, 0.30, 0.25, 0.15, 0.07, 0.03]
        ).astype(float),

        "months_since_recent_delinquency": np.where(
            rng.uniform(0, 1, n_declined) < 0.35, 999.0,
            rng.uniform(1, 48, n_declined)
        ),

        "num_inquiries_last_6m": rng.choice(
            [0, 1, 2, 3, 4, 5, 6], n_declined,
            p=[0.15, 0.20, 0.22, 0.18, 0.12, 0.08, 0.05]
        ).astype(float),

        "employment_length_years": rng.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_declined,
            p=[0.15, 0.12, 0.12, 0.10, 0.09, 0.09, 0.08, 0.08, 0.07, 0.06, 0.04]
        ).astype(float),

        "total_accounts": rng.integers(1, 20, n_declined).astype(float),

        "months_since_oldest_trade": rng.uniform(0, 180, n_declined),

        "loan_amount": np.exp(
            rng.normal(9.5, 0.8, n_declined)
        ).clip(1_000, 40_000),

        "alt_data_score": (
            40 + credit_quality * 25
            + rng.normal(0, 10, n_declined)
        ).clip(0, 100),

        "ltv_ratio": np.nan,
        "product_type": 0,          # Declined apps are unsecured
        "external_risk_estimate": -1.0,
        "pct_trades_never_delinquent": -1.0,
        "months_since_recent_trade": -1.0,
        "num_high_utilization_trades": -1.0,
        "loan_term_months": 36.0,
        "num_open_accounts": rng.integers(1, 15, n_declined).astype(float),
        "thin_file_flag": rng.choice([True, False], n_declined,
                                      p=[0.25, 0.75]),
        "data_source": "synthetic_declined",
        "origination_year": 2013,

        # No default flag yet — assigned by initial model
        "default_flag": np.nan,
    })

    # Fix ltv for unsecured
    declined["ltv_ratio"] = 0.0

    log.info(f"  Synthetic declined: avg credit score = "
             f"{declined['credit_score'].mean():.0f} "
             f"(vs {approved_score_mean:.0f} approved)")
    log.info(f"  Synthetic declined: avg DTI = "
             f"{declined['dti'].mean():.1f}% "
             f"(vs {approved_dti_mean:.1f}% approved)")

    return declined


def apply_reject_inference(
        df_train: pd.DataFrame,
        initial_model,
        feature_cols: list,
        n_declined: int = 5000,
        bad_rate_assumption: float = 0.55,
        random_state: int = 42) -> pd.DataFrame:
    """
    Apply reject inference augmentation to training data.

    Args:
        df_train: Original training data (approved loans).
        initial_model: Trained initial PD model (sklearn-compatible).
        feature_cols: Feature columns used by the model.
        n_declined: Number of synthetic declined records.
        bad_rate_assumption: Assumed default rate for declined population.
        random_state: Reproducibility seed.

    Returns:
        Augmented training DataFrame including re-weighted declined records.
    """
    log.info("Applying reject inference augmentation ...")
    rng = np.random.default_rng(random_state)

    # Generate declined population
    df_declined = generate_reject_inference(df_train, n_declined, random_state)

    # Add any interaction features that are in the training data
    for col in ["ltv_x_product", "dti_x_emp_risk",
                "ads_x_thin_file", "score_x_product"]:
        if col in df_train.columns and col not in df_declined.columns:
            df_declined[col] = 0.0

    # Score declined using initial model
    available_features = [f for f in feature_cols if f in df_declined.columns]
    X_declined = df_declined[available_features].fillna(0)

    try:
        pd_declined = initial_model.predict_proba(X_declined)[:, 1]
    except Exception as e:
        log.warning(f"Initial model scoring failed ({e}) — "
                    "using bad_rate_assumption for all declined")
        pd_declined = np.full(n_declined, bad_rate_assumption)

    # Calibrate to bad_rate_assumption
    # Scale raw scores so mean = bad_rate_assumption
    current_mean = pd_declined.mean()
    if current_mean > 0:
        pd_declined = pd_declined * (bad_rate_assumption / current_mean)
    pd_declined = np.clip(pd_declined, 0.01, 0.99)

    # Assign probabilistic labels
    df_declined["default_flag"] = (
        rng.uniform(0, 1, n_declined) < pd_declined
    ).astype(float)

    # Weight declined records
    # We want declined to represent ~40% of marginal credit decisions
    # Weight each record as 0.5 (half influence of an approved loan)
    df_declined["sample_weight"] = 0.5
    df_train_aug = df_train.copy()
    df_train_aug["sample_weight"] = 1.0

    df_augmented = pd.concat(
        [df_train_aug, df_declined], ignore_index=True
    )

    log.info(f"  Original training: {len(df_train):,} rows")
    log.info(f"  Added declined:    {len(df_declined):,} rows")
    log.info(f"  Augmented total:   {len(df_augmented):,} rows")
    log.info(f"  Declined default rate (assigned): "
             f"{df_declined['default_flag'].mean():.2%}")
    log.info(f"  Augmented overall default rate: "
             f"{df_augmented['default_flag'].mean():.2%}")

    return df_augmented


def compute_pit_to_ttc(pd_pit: np.ndarray,
                        long_run_average_pd: float = 0.08,
                        smoothing_factor: float = 0.3) -> np.ndarray:
    """
    Convert Point-in-Time (PIT) PD to Through-the-Cycle (TTC) PD.

    PIT PD: Current best estimate — used for IFRS 9 ECL provisioning.
    TTC PD: Long-run average — used for Basel III regulatory capital.

    Method: Exponential smoothing toward long-run average.
    The smoothing_factor controls how much the TTC is anchored to the
    long-run average vs the current PIT estimate.

    Args:
        pd_pit: Array of PIT PD estimates.
        long_run_average_pd: Historical average default rate (8% is typical
                              for mixed retail credit portfolios).
        smoothing_factor: Weight on long-run average (0 = pure PIT,
                          1 = pure long-run average). 0.3 is typical.

    Returns:
        TTC PD array — smoother, less cyclical than PIT.
    """
    pd_ttc = (
        (1 - smoothing_factor) * pd_pit
        + smoothing_factor * long_run_average_pd
    )
    return np.clip(pd_ttc, 1e-6, 1 - 1e-6)
