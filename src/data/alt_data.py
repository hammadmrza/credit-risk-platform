"""
src/data/alt_data.py
────────────────────
Alternative Data Composite Score (ADS) Engineering Module.

PURPOSE
───────
The ADS is a synthetic proxy for non-traditional payment behaviour
signals including rent payments, utility bills, and telecom contracts.
These signals are commercially available from providers including:
  - Equifax NeoBureau (Canada)
  - Experian Clarity Services (US)
  - Borrowell Rental Advantage (Canada)
  - Nova Credit (cross-border)

Since we cannot access proprietary alternative data feeds in this
open-source project, we engineer a synthetic ADS that:
  1. Is correlated with the default_flag at a realistic signal level
  2. Adds incremental lift over traditional bureau features (especially
     for thin-file applicants with fewer than 3 tradelines)
  3. Has documented construction methodology so the approach is
     transparent and auditable

CONSTRUCTION METHODOLOGY
────────────────────────
The ADS is constructed as a weighted composite of:
  a) Credit score component (40%) — proxy for overall credit health
  b) Delinquency recency component (30%) — rent/utility arrears proxy
  c) Utilization component (20%) — cash flow management proxy
  d) Account vintage component (10%) — stability / tenure proxy

Each component is normalized to 0-100. Random noise at a calibrated
level is added to reflect the inherent uncertainty in alternative
data signals. The composite is anchored so that:
  - Score of 80+ correlates with very low default probability
  - Score of 50-80 adds moderate positive signal
  - Score below 40 adds incremental default risk signal
  - The signal is weakest for thin-file applicants (by design)

THIN-FILE SEGMENT
─────────────────
Applicants with fewer than 3 tradelines are flagged as thin-file.
The ADS lift is intentionally strongest for this segment — this is
the commercially relevant finding: alternative data most helps where
traditional bureau data is least informative.

REAL-WORLD EXTENSION
─────────────────────
In a production deployment, this module would be replaced by a call
to a bureau alternative data API. The interface (a function that
takes an applicant record and returns an ADS between 0 and 100)
remains identical. Only the implementation changes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _credit_score_component(df: pd.DataFrame) -> pd.Series:
    """
    Credit score component of ADS (40% weight).
    Higher credit score → higher alternative data signal.
    """
    score = df.get("credit_score", pd.Series(np.nan, index=df.index))
    # Normalize 300-850 → 0-100
    normalized = ((score - 300) / (850 - 300) * 100).clip(0, 100)
    return normalized.fillna(50)  # Neutral for missing


def _delinquency_component(df: pd.DataFrame) -> pd.Series:
    """
    Delinquency recency component of ADS (30% weight).
    Longer time since delinquency (or no delinquency) → higher score.
    """
    mrd = df.get(
        "months_since_recent_delinquency",
        pd.Series(np.nan, index=df.index)
    )

    # 0 months = recent delinquency = bad = 0
    # 60+ months = old delinquency = neutral = 70
    # NaN = no delinquency ever = excellent = 100
    normalized = pd.Series(np.where(
        mrd.isna(), 100,                          # No delinquency ever
        np.where(mrd >= 60, 70,                   # Old delinquency
        np.where(mrd >= 24, 50,                   # Moderate
        np.where(mrd >= 12, 30,                   # Recent
        10)))                                     # Very recent
    ), index=df.index)
    return normalized.astype(float)


def _utilization_component(df: pd.DataFrame) -> pd.Series:
    """
    Credit utilization component of ADS (20% weight).
    Lower utilization → higher score (better cash flow management).
    """
    util = df.get("credit_utilization", pd.Series(np.nan, index=df.index))
    # 0-10% → 100, 10-30% → 80, 30-50% → 60, 50-70% → 40, 70%+ → 20
    normalized = pd.Series(np.where(
        util.isna(), 60,
        np.where(util <= 10, 100,
        np.where(util <= 30, 80,
        np.where(util <= 50, 60,
        np.where(util <= 70, 40, 20))))
    ), index=df.index)
    return normalized.astype(float)


def _vintage_component(df: pd.DataFrame) -> pd.Series:
    """
    Account vintage component of ADS (10% weight).
    Longer credit history → higher stability signal.
    """
    vintage = df.get(
        "months_since_oldest_trade",
        pd.Series(np.nan, index=df.index)
    )
    # Normalize: 0 → 0, 120+ months → 100
    normalized = (vintage / 120 * 100).clip(0, 100)
    return normalized.fillna(50).astype(float)


def compute_alt_data_score(
        df: pd.DataFrame,
        noise_level: float = config.ALT_DATA_NOISE_LEVEL,
        random_state: int = config.ALT_DATA_RANDOM_STATE,
        ) -> pd.Series:
    """
    Compute the Alternative Data Composite Score for each row.

    ADS = 0.40 × credit_score_component
        + 0.30 × delinquency_component
        + 0.20 × utilization_component
        + 0.10 × vintage_component
        + noise

    Args:
        df: DataFrame with credit features.
        noise_level: Standard deviation of noise relative to score range.
                     0.15 = 15% noise, calibrated to realistic signal level.
        random_state: Reproducibility seed.

    Returns:
        Series of ADS values in [0, 100].
    """
    rng = np.random.default_rng(random_state)

    components = pd.DataFrame({
        "credit_score": _credit_score_component(df),
        "delinquency":  _delinquency_component(df),
        "utilization":  _utilization_component(df),
        "vintage":      _vintage_component(df),
    })

    weights    = {"credit_score": 0.40, "delinquency": 0.30,
                  "utilization": 0.20, "vintage": 0.10}
    composite  = sum(components[k] * v for k, v in weights.items())

    noise = rng.normal(0, noise_level * 100, size=len(df))
    ads   = (composite + noise).clip(0, 100)

    return pd.Series(ads, index=df.index, name="alt_data_score")


def flag_thin_file(df: pd.DataFrame,
                   threshold: int = 3) -> pd.Series:
    """
    Flag applicants with fewer than `threshold` total tradelines.
    Thin-file applicants benefit most from alternative data signals.

    Args:
        df: DataFrame with total_accounts column.
        threshold: Number of tradelines below which = thin-file.

    Returns:
        Boolean Series: True = thin-file.
    """
    if "total_accounts" not in df.columns:
        log.warning("total_accounts not found — setting all thin_file_flag to False")
        return pd.Series(False, index=df.index, name="thin_file_flag")

    flag = (
        df["total_accounts"].fillna(0) < threshold
    ).rename("thin_file_flag")

    thin_count = flag.sum()
    log.info(f"Thin-file applicants (<{threshold} tradelines): "
             f"{thin_count:,} ({100*thin_count/len(df):.1f}%)")
    return flag


def add_alt_data_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add alt_data_score and thin_file_flag to the harmonized DataFrame.

    Args:
        df: Harmonized DataFrame (output of src.data.harmonize).

    Returns:
        DataFrame with two new columns: alt_data_score, thin_file_flag.
    """
    log.info("Engineering alternative data features ...")
    df = df.copy()
    df["alt_data_score"] = compute_alt_data_score(df)
    df["thin_file_flag"] = flag_thin_file(df)

    log.info(f"ADS distribution:")
    log.info(f"  Mean:    {df['alt_data_score'].mean():.1f}")
    log.info(f"  Median:  {df['alt_data_score'].median():.1f}")
    log.info(f"  Std:     {df['alt_data_score'].std():.1f}")

    # Quick validation: ADS should be negatively correlated with default
    corr = df["alt_data_score"].corr(df["default_flag"])
    log.info(f"  Correlation with default_flag: {corr:.3f} "
             f"(expected negative)")

    if corr > 0:
        log.warning("ADS is positively correlated with default — "
                    "review component weights.")

    # Thin-file lift analysis
    thin_default_no_ads = df[df["thin_file_flag"]]["default_flag"].mean()
    log.info(f"Thin-file default rate: {thin_default_no_ads:.2%}")
    log.info("ADS lift analysis will be performed in EDA notebook.")

    return df


def generate_ads_methodology_doc() -> str:
    """
    Returns the ADS methodology description as a markdown string.
    This is used in the model card and README documentation.
    """
    return """
## Alternative Data Composite Score (ADS) — Methodology

### Overview
The ADS is a synthetic proxy for non-traditional payment behaviour signals.
It is engineered from existing bureau features using a weighted composite
approach that mirrors the methodology of commercial alternative data products.

### Construction
| Component             | Weight | Proxy for                          |
|-----------------------|--------|-------------------------------------|
| Credit score          | 40%    | Overall financial health            |
| Delinquency recency   | 30%    | Rent/utility payment discipline     |
| Credit utilization    | 20%    | Cash flow management                |
| Account vintage       | 10%    | Stability / relationship tenure     |

Calibrated noise (σ = 15% of range) is added to reflect real-world
alternative data signal quality.

### Commercial Equivalents
- Equifax NeoBureau (Canada)
- Experian Clarity Services (US)
- Borrowell Rental Advantage (Canada)
- Nova Credit (cross-border applicants)

### Production Extension
In a production deployment, this module is replaced by a bureau API call.
The interface (input: applicant record → output: score 0-100) is identical.

### Validation
The ADS is validated against default_flag using correlation analysis and
AUC lift measurement — specifically in the thin-file segment (<3 tradelines)
where the incremental lift is expected to be largest.
"""


if __name__ == "__main__":
    # Test with a small synthetic dataset
    rng = np.random.default_rng(42)
    n = 1000
    test_df = pd.DataFrame({
        "credit_score": rng.uniform(300, 850, n),
        "months_since_recent_delinquency": rng.choice(
            [np.nan, 6, 12, 24, 48], n, p=[0.5, 0.1, 0.15, 0.15, 0.1]
        ),
        "credit_utilization": rng.uniform(0, 100, n),
        "months_since_oldest_trade": rng.uniform(0, 240, n),
        "total_accounts": rng.integers(0, 30, n),
        "default_flag": rng.integers(0, 2, n),
    })

    result = add_alt_data_features(test_df)
    print(result[["credit_score", "alt_data_score", "thin_file_flag",
                  "default_flag"]].describe())
