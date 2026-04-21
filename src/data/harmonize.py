"""
src/data/harmonize.py
─────────────────────
Harmonization module: combines LendingClub (unsecured) and FICO HELOC
(secured) datasets into a single unified DataFrame for modelling.

Design decisions documented here:
- The shared feature schema captures bureau-adjacent signals present
  in both datasets. Where a feature exists natively in one dataset
  but is synthetic in the other, this is flagged in the FEATURE_PROVENANCE
  dictionary below and in the has_synthetic_features column.
- The product_type flag (0=unsecured, 1=secured) is the key segmentation
  variable. All downstream models use it as a feature and as a conditioning
  variable for SHAP analysis.
- The out-of-time split is temporal: train on origination years ≤ 2015,
  test on ≥ 2016. This prevents data leakage and mirrors real validation
  practice (OSFI E-23, Basel II/III model validation standards).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Shared feature schema ────────────────────────────────────────
# These are the features that appear in the harmonized dataset.
# Source column: N = native (exists in raw data), S = synthetic/engineered
SHARED_FEATURES = [
    # Core application features
    "loan_amount",                  # LC: N,  HELOC: S (from credit line range)
    "annual_income",                # LC: N,  HELOC: S (from loan_amount + DTI proxy)
    "dti",                          # LC: N,  HELOC: S (from revolving/installment burden)
    "credit_score",                 # LC: N,  HELOC: S (scaled ExternalRiskEstimate)
    "employment_length_years",      # LC: N,  HELOC: S (distribution from workforce data)
    "loan_term_months",             # LC: N,  HELOC: S (24 = HELOC performance window)

    # Bureau features — native in both
    "months_since_oldest_trade",    # LC: N,  HELOC: N (MSinceOldestTradeOpen)
    "months_since_recent_delinquency",  # LC: N,  HELOC: N (MSinceMostRecentDelq)
    "num_inquiries_last_6m",        # LC: N,  HELOC: N (NumInqLast6M)
    "num_derogatory_marks",         # LC: N,  HELOC: N (NumTrades90Ever2DerogPubRec)
    "total_accounts",               # LC: N,  HELOC: N (NumTotalTrades)
    "credit_utilization",           # LC: N,  HELOC: S (from revolving_burden)
    "num_open_accounts",            # LC: N,  HELOC: S (num_trades_open_last_12m proxy)

    # HELOC-specific bureau features (NaN for LC)
    "external_risk_estimate",       # LC: NaN, HELOC: N (raw ExternalRiskEstimate)
    "pct_trades_never_delinquent",  # LC: NaN, HELOC: N
    "months_since_recent_trade",    # LC: NaN, HELOC: N
    "num_high_utilization_trades",  # LC: NaN, HELOC: N

    # Secured-only features
    "ltv_ratio",                    # LC: NaN, HELOC: S (engineered, corr. with credit score)

    # Alternative data composite (engineered for both)
    "alt_data_score",               # Engineered: synthetic proxy for rent/util/telco

    # Model flags
    "product_type",                 # 0=unsecured (LC), 1=secured (HELOC)
    "has_synthetic_features",       # Transparency: True if synthetic fields used

    # Temporal fields
    "origination_date",
    "origination_year",
    "origination_quarter",

    # Target
    "default_flag",
]

# Categorical features (will be one-hot encoded or label encoded later)
CATEGORICAL_FEATURES = [
    "home_ownership",
    "purpose",
    "verification_status",
    "lc_grade",
    "addr_state",
]

# All numeric features used in modelling
NUMERIC_FEATURES = [
    "loan_amount", "annual_income", "dti", "credit_score",
    "employment_length_years", "loan_term_months",
    "months_since_oldest_trade", "months_since_recent_delinquency",
    "num_inquiries_last_6m", "num_derogatory_marks", "total_accounts",
    "credit_utilization", "num_open_accounts", "external_risk_estimate",
    "pct_trades_never_delinquent", "months_since_recent_trade",
    "num_high_utilization_trades", "ltv_ratio", "alt_data_score",
    "product_type",
]


def _align_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Ensure a dataset has all expected columns, filling missing ones with NaN.

    Args:
        df: Input DataFrame.
        dataset_name: Name for logging.

    Returns:
        DataFrame with all SHARED_FEATURES columns present.
    """
    all_expected = (
        SHARED_FEATURES +
        [c for c in CATEGORICAL_FEATURES if c not in SHARED_FEATURES]
    )

    missing = [c for c in all_expected if c not in df.columns]
    if missing:
        log.info(f"{dataset_name}: Adding {len(missing)} missing columns as NaN: "
                 f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        for col in missing:
            df[col] = np.nan

    # Also add loan_term_months if missing (HELOC doesn't have it natively)
    if "loan_term_months" not in df.columns:
        df["loan_term_months"] = 24.0  # HELOC 24-month performance window

    return df


def harmonize(df_lc: pd.DataFrame,
              df_heloc: pd.DataFrame) -> pd.DataFrame:
    """
    Combine LendingClub and HELOC datasets into a single harmonized DataFrame.

    Args:
        df_lc: Cleaned LendingClub DataFrame (from src.data.lendingclub).
        df_heloc: Cleaned HELOC DataFrame (from src.data.heloc).

    Returns:
        Unified DataFrame with product_type flag and harmonized schema.
    """
    log.info("Harmonizing datasets ...")
    log.info(f"  LendingClub: {len(df_lc):,} rows")
    log.info(f"  HELOC:       {len(df_heloc):,} rows")

    # Align columns
    df_lc_aligned    = _align_columns(df_lc.copy(),    "LendingClub")
    df_heloc_aligned = _align_columns(df_heloc.copy(), "HELOC")

    # Assign dataset source for provenance tracking
    df_lc_aligned["data_source"]    = "lendingclub"
    df_heloc_aligned["data_source"] = "heloc"

    # Select only the harmonized columns (plus source tracking)
    keep_cols = list(dict.fromkeys(
        SHARED_FEATURES +
        [c for c in CATEGORICAL_FEATURES if c in df_lc_aligned.columns] +
        ["data_source"]
    ))

    lc_keep    = [c for c in keep_cols if c in df_lc_aligned.columns]
    heloc_keep = [c for c in keep_cols if c in df_heloc_aligned.columns]

    df = pd.concat([
        df_lc_aligned[lc_keep],
        df_heloc_aligned[heloc_keep]
    ], ignore_index=True, sort=False)

    log.info(f"Combined dataset: {len(df):,} rows, {df.shape[1]} columns")
    log.info(f"  Unsecured (LendingClub): {(df['product_type']==0).sum():,}")
    log.info(f"  Secured (HELOC):         {(df['product_type']==1).sum():,}")
    log.info(f"  Overall default rate:    {df['default_flag'].mean():.2%}")
    log.info(f"  Unsecured default rate:  {df[df['product_type']==0]['default_flag'].mean():.2%}")
    log.info(f"  Secured default rate:    {df[df['product_type']==1]['default_flag'].mean():.2%}")

    return df


def create_oot_split(df: pd.DataFrame,
                     train_cutoff_year: int = config.LC_TRAIN_CUTOFF_YEAR
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create an out-of-time (OOT) train/test split.

    The HELOC dataset is cross-sectional (no origination date) and is
    split randomly 80/20 to maintain its proportion in both sets.
    LendingClub is split temporally.

    Args:
        df: Harmonized DataFrame.
        train_cutoff_year: LendingClub loans from this year onward go to test.

    Returns:
        (df_train, df_test) tuple.
    """
    log.info(f"Creating OOT split (LC cutoff year: {train_cutoff_year}) ...")

    # ── LendingClub: temporal split ──────────────────────────────
    lc_mask  = df["data_source"] == "lendingclub"
    df_lc    = df[lc_mask].copy()
    df_heloc = df[~lc_mask].copy()

    lc_train_mask  = df_lc["origination_year"] < train_cutoff_year
    df_lc_train    = df_lc[lc_train_mask]
    df_lc_test     = df_lc[~lc_train_mask]

    log.info(f"  LC train vintages:  {df_lc_train['origination_year'].min():.0f}–"
             f"{df_lc_train['origination_year'].max():.0f}  "
             f"({len(df_lc_train):,} rows)")
    log.info(f"  LC test vintages:   {df_lc_test['origination_year'].min():.0f}–"
             f"{df_lc_test['origination_year'].max():.0f}  "
             f"({len(df_lc_test):,} rows)")

    # ── HELOC: random 80/20 split ─────────────────────────────────
    from sklearn.model_selection import train_test_split
    heloc_train, heloc_test = train_test_split(
        df_heloc,
        test_size=0.20,
        stratify=df_heloc["default_flag"],
        random_state=config.LC_RANDOM_STATE
    )

    log.info(f"  HELOC train: {len(heloc_train):,} rows")
    log.info(f"  HELOC test:  {len(heloc_test):,} rows")

    # ── Combine ───────────────────────────────────────────────────
    df_train = pd.concat([df_lc_train, heloc_train],
                         ignore_index=True).sample(
        frac=1, random_state=config.LC_RANDOM_STATE
    ).reset_index(drop=True)

    df_test = pd.concat([df_lc_test, heloc_test],
                        ignore_index=True).sample(
        frac=1, random_state=config.LC_RANDOM_STATE
    ).reset_index(drop=True)

    log.info(f"Final train set: {len(df_train):,} rows | "
             f"default rate: {df_train['default_flag'].mean():.2%}")
    log.info(f"Final test set:  {len(df_test):,} rows  | "
             f"default rate: {df_test['default_flag'].mean():.2%}")

    return df_train, df_test


def run_harmonization(save: bool = True
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full harmonization pipeline.

    Loads cleaned LendingClub and HELOC parquet files, combines them,
    creates OOT split, and optionally saves all three outputs.

    Args:
        save: Whether to save processed files.

    Returns:
        (df_harmonized, df_train, df_test) tuple.
    """
    if not config.LC_CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"{config.LC_CLEAN_PATH} not found. "
            "Run src/data/lendingclub.py first."
        )
    if not config.HELOC_CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"{config.HELOC_CLEAN_PATH} not found. "
            "Run src/data/heloc.py first."
        )

    df_lc    = pd.read_parquet(config.LC_CLEAN_PATH)
    df_heloc = pd.read_parquet(config.HELOC_CLEAN_PATH)

    df_harmonized        = harmonize(df_lc, df_heloc)
    df_train, df_test    = create_oot_split(df_harmonized)

    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_harmonized.to_parquet(config.HARMONIZED_PATH, index=False)
        df_train.to_parquet(config.TRAIN_PATH, index=False)
        df_test.to_parquet(config.TEST_PATH, index=False)
        log.info(f"Saved: {config.HARMONIZED_PATH}")
        log.info(f"Saved: {config.TRAIN_PATH}")
        log.info(f"Saved: {config.TEST_PATH}")

    return df_harmonized, df_train, df_test


if __name__ == "__main__":
    df, train, test = run_harmonization()
    print(df.dtypes)
    print(df.describe())
