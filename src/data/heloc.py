"""
src/data/heloc.py
─────────────────
FICO HELOC data acquisition and cleaning module.

The HELOC dataset is a bureau-only dataset — all 23 features are
derived from credit bureau tradeline data. It does not include
income, employment, or collateral fields directly. Where the
harmonized schema requires these, we engineer synthetic proxies
from the available bureau features with documented assumptions.

Missing values in HELOC are encoded as special codes:
  -7 → condition not applicable
  -8 → no usable or valid trades
  -9 → no record (equivalent to null)

All three are treated as NaN in the processed output.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── HELOC column → harmonized name mapping ───────────────────────
HELOC_RENAME = {
    "ExternalRiskEstimate":               "external_risk_estimate",
    "MSinceOldestTradeOpen":              "months_since_oldest_trade",
    "MSinceMostRecentTradeOpen":          "months_since_recent_trade",
    "AverageMInFile":                     "avg_months_in_file",
    "NumSatisfactoryTrades":              "num_satisfactory_trades",
    "NumTrades60Ever2DerogPubRec":        "num_trades_60dpd",
    "NumTrades90Ever2DerogPubRec":        "num_derogatory_marks",
    "PercentTradesNeverDelq":             "pct_trades_never_delinquent",
    "MSinceMostRecentDelq":               "months_since_recent_delinquency",
    "MaxDelq2PublicRecLast12M":           "max_delinquency_last_12m",
    "MaxDelqEver":                        "max_delinquency_ever",
    "NumTotalTrades":                     "total_accounts",
    "NumTradesOpeninLast12M":             "num_trades_open_last_12m",
    "PercentInstallTrades":               "pct_installment_trades",
    "MSinceMostRecentInqexcl7days":       "months_since_recent_inquiry",
    "NumInqLast6M":                       "num_inquiries_last_6m",
    "NumInqLast6Mexcl7days":              "num_inquiries_last_6m_excl7",
    "NetFractionRevolvingBurden":         "revolving_burden",
    "NetFractionInstallBurden":           "installment_burden",
    "NumRevolvingTradesWBalance":         "num_revolving_trades_w_balance",
    "NumInstallTradesWBalance":           "num_installment_trades_w_balance",
    "NumBank2NatlTradesWHighUtilization": "num_high_utilization_trades",
    "PercentTradesWBalance":              "pct_trades_with_balance",
}


def load_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw HELOC CSV.

    Args:
        path: Path to heloc_dataset_v1.csv. Defaults to config.HELOC_RAW_PATH.

    Returns:
        Raw HELOC DataFrame.
    """
    path = path or config.HELOC_RAW_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"HELOC raw file not found at {path}\n"
            "See data/README.md for download instructions.\n"
            "Or run: python src/data/sample_generator.py"
        )

    log.info(f"Loading HELOC data from {path} ...")
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the raw HELOC DataFrame.

    Steps:
    1. Encode target variable
    2. Replace HELOC missing codes (-7, -8, -9) with NaN
    3. Rename columns to harmonized schema
    4. Engineer synthetic proxies for fields not in bureau data
       (credit_score, dti, annual_income, loan_amount, ltv_ratio)
    5. Add product type flag

    Args:
        df: Raw HELOC DataFrame (output of load_raw).

    Returns:
        Cleaned DataFrame aligned to harmonized schema.
    """
    log.info("Cleaning HELOC data ...")
    df = df.copy()

    # ── 1. Target variable ────────────────────────────────────────
    df["default_flag"] = (
        df["RiskPerformance"] == config.HELOC_BAD_VALUE
    ).astype(int)

    # ── 2. Replace HELOC missing codes with NaN ───────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df.loc[df[col].isin(config.HELOC_MISSING_CODES), col] = np.nan

    # ── 3. Rename columns ─────────────────────────────────────────
    df = df.rename(columns=HELOC_RENAME)

    # ── 4. Engineer synthetic proxies ────────────────────────────
    # These are documented assumptions — clearly labelled as synthetic
    # in the harmonized dataset via the _synthetic suffix in provenance.

    # credit_score: ExternalRiskEstimate is a third-party risk score
    # on an arbitrary scale. We scale it linearly to 300-850.
    # ExternalRiskEstimate range in data is approximately 0-100.
    if "external_risk_estimate" in df.columns:
        est = df["external_risk_estimate"]
        est_min, est_max = 0, 100
        df["credit_score"] = (
            300 + (est - est_min) / (est_max - est_min) * (850 - 300)
        ).clip(300, 850)

    # loan_amount: HELOC credit lines range $5,000-$150,000 per FICO docs.
    # We generate a realistic distribution correlated with credit quality.
    rng = np.random.default_rng(config.ALT_DATA_RANDOM_STATE)
    n = len(df)

    # Higher credit scores → higher credit limits (realistic)
    if "credit_score" in df.columns:
        score_norm = (df["credit_score"] - 300) / (850 - 300)
        log_amount = 8.5 + score_norm * 2.0 + rng.normal(0, 0.4, n)
        df["loan_amount"] = np.exp(log_amount).clip(5_000, 150_000).round(-2)
    else:
        df["loan_amount"] = np.exp(rng.normal(10.5, 0.8, n)).clip(5_000, 150_000)

    # annual_income: Inferred from loan amount and a typical HELOC
    # debt-service ratio assumption (annual payment ~8% of income).
    # This is a synthetic proxy — documented assumption.
    df["annual_income"] = (df["loan_amount"] * 0.08 * 12 /
                           rng.uniform(0.25, 0.45, n)).clip(20_000, 500_000).round(-2)

    # dti: Estimated from revolving_burden and installment_burden.
    # revolving_burden = revolving balance / revolving limit (as pct)
    # installment_burden = installment balance / original amount (as pct)
    # Combined, these proxy total debt service relative to income.
    if "revolving_burden" in df.columns and "installment_burden" in df.columns:
        rb = df["revolving_burden"].fillna(df["revolving_burden"].median())
        ib = df["installment_burden"].fillna(df["installment_burden"].median())
        df["dti"] = ((rb * 0.4 + ib * 0.6) / 3).clip(0, 80)
    else:
        df["dti"] = rng.uniform(10, 45, n)

    # ltv_ratio: HELOC is secured by home equity.
    # Typical HELOC LTV at origination is 60-85%.
    # We generate realistic LTV correlated with credit risk:
    # higher-risk borrowers tend to have higher LTV.
    if "credit_score" in df.columns:
        score_norm = (df["credit_score"] - 300) / (850 - 300)
        # Better scores → lower LTV (more equity cushion)
        ltv_base = 0.85 - score_norm * 0.25
        df["ltv_ratio"] = (
            ltv_base + rng.normal(0, 0.05, n)
        ).clip(0.40, 0.95)
    else:
        df["ltv_ratio"] = rng.uniform(0.55, 0.90, n)

    # employment_length_years: Not in bureau data. Generate from
    # a distribution calibrated to US workforce statistics.
    emp_choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    emp_weights = [0.08, 0.08, 0.09, 0.09, 0.09, 0.10, 0.09, 0.09, 0.09, 0.09, 0.11]
    df["employment_length_years"] = rng.choice(
        emp_choices, size=n, p=emp_weights
    )

    # credit_utilization: Derive from revolving_burden if available
    if "revolving_burden" in df.columns:
        df["credit_utilization"] = df["revolving_burden"].clip(0, 100)
    else:
        df["credit_utilization"] = rng.uniform(10, 80, n)

    # num_derogatory_marks: Use num_trades_60dpd as proxy if pub_rec not available
    if "num_derogatory_marks" not in df.columns and "num_trades_60dpd" in df.columns:
        df["num_derogatory_marks"] = df["num_trades_60dpd"]

    # ── 5. Add product type and origination placeholders ─────────
    df["product_type"]       = config.PRODUCT_TYPE_SECURED
    df["origination_year"]   = 2017    # HELOC dataset is cross-sectional, no date
    df["origination_quarter"] = "2017Q2"
    df["origination_date"]   = pd.Timestamp("2017-06-01")
    df["lc_grade"]           = np.nan  # Not applicable for HELOC

    # ── 6. Mark synthetic columns ─────────────────────────────────
    df["has_synthetic_features"] = True  # Transparency flag

    log.info(f"Cleaning complete. Shape: {df.shape}")
    log.info(f"Default rate: {df['default_flag'].mean():.2%}")

    _describe_synthetic_fields(df)
    return df


def _describe_synthetic_fields(df: pd.DataFrame) -> None:
    """Log a summary of which fields were engineered synthetically."""
    synthetic_fields = [
        "loan_amount", "annual_income", "dti",
        "ltv_ratio", "employment_length_years", "credit_utilization",
        "origination_date", "origination_year"
    ]
    log.info("Synthetic field summary (HELOC-specific engineering):")
    for f in synthetic_fields:
        if f in df.columns:
            log.info(f"  {f:35s}: mean={df[f].mean():.2f}, "
                     f"null={df[f].isna().sum()}")


def load_and_process(path: Optional[Path] = None,
                     save: bool = True) -> pd.DataFrame:
    """
    Full pipeline: load → clean → save.

    Args:
        path: Path to raw CSV. Defaults to config.HELOC_RAW_PATH.
        save: Whether to save the processed file.

    Returns:
        Cleaned HELOC DataFrame.
    """
    df_raw   = load_raw(path)
    df_clean = clean(df_raw)

    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_clean.to_parquet(config.HELOC_CLEAN_PATH, index=False)
        log.info(f"Saved to {config.HELOC_CLEAN_PATH}")

    return df_clean


if __name__ == "__main__":
    df = load_and_process()
    print(df.head())
    print(df.dtypes)
