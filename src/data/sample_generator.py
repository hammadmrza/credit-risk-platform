"""
src/data/sample_generator.py
─────────────────────────────
Generates realistic synthetic datasets at the correct file paths
so the entire pipeline can run end-to-end without downloading
the real LendingClub and HELOC datasets.

The synthetic data:
- Follows the same schema as the real datasets
- Has realistic feature distributions and correlations
- Preserves a realistic default rate (~20% for LC, ~35% for HELOC)
- Is NOT a substitute for the real data for model development
- IS sufficient for running and testing all code

Run: python src/data/sample_generator.py
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

LC_N    = 50_000    # Smaller than real for testing speed
HELOC_N = 10_459    # Same as real HELOC


def generate_lendingclub(n: int = LC_N,
                          random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic LendingClub-like dataset."""
    log.info(f"Generating synthetic LendingClub data ({n:,} rows) ...")
    rng = np.random.default_rng(random_state)

    # Base credit quality
    credit_quality = rng.beta(4, 2, n)  # Skewed toward good borrowers

    fico_low  = (300 + credit_quality * 500 + rng.normal(0, 20, n)).clip(300, 840).round()
    fico_high = (fico_low + rng.integers(10, 30, n)).clip(310, 850)
    dti       = (50 - credit_quality * 35 + rng.normal(0, 5, n)).clip(0, 80)
    annual_inc = np.exp(10.5 + credit_quality * 1.5 + rng.normal(0, 0.4, n)).clip(15000, 500000)

    # Grade based on credit quality
    grades = pd.cut(
        credit_quality,
        bins=[0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0],
        labels=["G", "F", "E", "D", "C", "B"]
    ).astype(str)
    grades = np.where(credit_quality > 0.90, "A", grades)

    # Origination dates (2007-2018)
    years   = rng.integers(2007, 2019, n)
    months  = rng.integers(1, 13, n)
    issue_d = [f"{pd.Timestamp(y, m, 1).strftime('%b-%Y')}"
               for y, m in zip(years, months)]

    # Default flag: inversely related to credit quality
    default_prob = 0.55 - credit_quality * 0.50
    default_flag = (rng.uniform(0, 1, n) < default_prob).astype(int)

    df = pd.DataFrame({
        "loan_amnt":           rng.integers(1000, 40001, n),
        "term":                rng.choice([" 36 months", " 60 months"], n,
                                           p=[0.70, 0.30]),
        "int_rate":            (6 + (1 - credit_quality) * 25 +
                                rng.normal(0, 1, n)).clip(5, 31),
        "installment":         rng.uniform(50, 1500, n).round(2),
        "grade":               grades,
        "sub_grade":           [f"{g}{rng.integers(1,6)}"
                                for g in grades],
        "purpose":             rng.choice(
            ["debt_consolidation", "credit_card", "home_improvement",
             "other", "major_purchase", "medical", "car", "vacation"],
            n, p=[0.45, 0.20, 0.10, 0.10, 0.07, 0.04, 0.03, 0.01]
        ),
        "emp_length":          rng.choice(
            ["< 1 year", "1 year", "2 years", "3 years", "4 years",
             "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
            n, p=[0.08, 0.08, 0.09, 0.09, 0.09, 0.10, 0.09, 0.09, 0.09, 0.09, 0.11]
        ),
        "home_ownership":      rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"],
                                           n, p=[0.48, 0.42, 0.08, 0.02]),
        "annual_inc":          annual_inc.round(2),
        "verification_status": rng.choice(
            ["Not Verified", "Source Verified", "Verified"],
            n, p=[0.40, 0.35, 0.25]
        ),
        "dti":                 dti.round(2),
        "delinq_2yrs":         rng.choice([0, 1, 2, 3], n, p=[0.78, 0.14, 0.05, 0.03]),
        "inq_last_6mths":      rng.choice([0, 1, 2, 3, 4, 5], n,
                                           p=[0.35, 0.28, 0.18, 0.10, 0.06, 0.03]),
        "mths_since_last_delinq": np.where(
            rng.uniform(0, 1, n) < 0.55, np.nan,
            rng.integers(1, 120, n).astype(float)
        ),
        "open_acc":            rng.integers(1, 30, n),
        "pub_rec":             rng.choice([0, 1, 2], n, p=[0.85, 0.12, 0.03]),
        "revol_bal":           np.exp(rng.normal(8.5, 1.2, n)).clip(0, 200000).round(2),
        "revol_util":          (rng.beta(2, 3, n) * 100).round(1),
        "total_acc":           rng.integers(2, 50, n),
        "mort_acc":            rng.choice([0, 1, 2, 3], n, p=[0.55, 0.25, 0.15, 0.05]),
        "pub_rec_bankruptcies": rng.choice([0, 1], n, p=[0.93, 0.07]),
        "fico_range_low":      fico_low,
        "fico_range_high":     fico_high,
        "earliest_cr_line":    [f"{pd.Timestamp(rng.integers(1970, 2010),
                                  rng.integers(1, 13), 1).strftime('%b-%Y')}"
                                for _ in range(n)],
        "loan_status":         np.where(default_flag == 1, "Charged Off", "Fully Paid"),
        "issue_d":             issue_d,
        "addr_state":          rng.choice(
            ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"],
            n, p=[0.15, 0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.09, 0.09, 0.06]
        ),
    })

    log.info(f"Generated LC sample: default rate = {default_flag.mean():.2%}")
    return df


def generate_heloc(n: int = HELOC_N,
                    random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic HELOC-like dataset matching FICO schema."""
    log.info(f"Generating synthetic HELOC data ({n:,} rows) ...")
    rng = np.random.default_rng(random_state + 1)

    credit_quality = rng.beta(3, 3, n)

    ext_risk     = (20 + credit_quality * 75 + rng.normal(0, 5, n)).clip(0, 100).round()
    default_prob = 0.75 - credit_quality * 0.70
    default_flag = np.where(
        rng.uniform(0, 1, n) < default_prob, "Bad", "Good"
    )

    def bureau_feature(base, noise=3, min_val=0, max_val=None):
        vals = (base + rng.normal(0, noise, n)).clip(min_val)
        if max_val:
            vals = vals.clip(max_val)
        return vals.round().astype(int)

    missing_mask = rng.uniform(0, 1, (n, 5)) < 0.08

    df = pd.DataFrame({
        "RiskPerformance":                    default_flag,
        "ExternalRiskEstimate":               ext_risk,
        "MSinceOldestTradeOpen":              bureau_feature(credit_quality * 250, 30, 0),
        "MSinceMostRecentTradeOpen":          bureau_feature(12, 8, 0),
        "AverageMInFile":                     bureau_feature(credit_quality * 150, 20, 0),
        "NumSatisfactoryTrades":              bureau_feature(credit_quality * 20, 4, 0),
        "NumTrades60Ever2DerogPubRec":        bureau_feature((1-credit_quality)*5, 2, 0),
        "NumTrades90Ever2DerogPubRec":        bureau_feature((1-credit_quality)*3, 1, 0),
        "PercentTradesNeverDelq":             (credit_quality * 100 +
                                              rng.normal(0, 8, n)).clip(0, 100).round(),
        "MSinceMostRecentDelq":               np.where(
            rng.uniform(0, 1, n) < 0.50, -7,
            bureau_feature(credit_quality * 60, 15, 0)
        ),
        "MaxDelq2PublicRecLast12M":           rng.choice([0, 1, 2, 3, 4, 5, 6, 7], n,
                                              p=[0.45, 0.20, 0.15, 0.08, 0.05, 0.04, 0.02, 0.01]),
        "MaxDelqEver":                        rng.choice([2, 3, 4, 5, 6, 7, 8], n,
                                              p=[0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.10]),
        "NumTotalTrades":                     bureau_feature(credit_quality * 30, 6, 1),
        "NumTradesOpeninLast12M":             bureau_feature(2, 2, 0),
        "PercentInstallTrades":               (rng.beta(2, 2, n) * 100).round().astype(int),
        "MSinceMostRecentInqexcl7days":       np.where(
            rng.uniform(0, 1, n) < 0.20, -7,
            bureau_feature(8, 6, 0)
        ),
        "NumInqLast6M":                       bureau_feature((1-credit_quality)*4, 2, 0),
        "NumInqLast6Mexcl7days":              bureau_feature((1-credit_quality)*3, 2, 0),
        "NetFractionRevolvingBurden":         ((1-credit_quality)*80 +
                                              rng.normal(0, 10, n)).clip(0, 100).round().astype(int),
        "NetFractionInstallBurden":           (rng.beta(2, 3, n) * 80).round().astype(int),
        "NumRevolvingTradesWBalance":         bureau_feature(credit_quality * 8, 2, 0),
        "NumInstallTradesWBalance":           bureau_feature(credit_quality * 5, 2, 0),
        "NumBank2NatlTradesWHighUtilization": bureau_feature((1-credit_quality)*4, 2, 0),
        "PercentTradesWBalance":              (credit_quality * 70 +
                                              rng.normal(0, 10, n)).clip(0, 100).round().astype(int),
    })

    log.info(f"Generated HELOC sample: default rate = {(default_flag=='Bad').mean():.2%}")
    return df


def generate_all(lc_n: int = LC_N,
                  heloc_n: int = HELOC_N) -> None:
    """
    Generate both synthetic datasets and save to config.RAW_DIR.
    """
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    df_lc = generate_lendingclub(lc_n)
    lc_path = config.LC_RAW_PATH
    df_lc.to_csv(lc_path, index=False)
    log.info(f"Saved synthetic LC data to {lc_path}")

    df_heloc = generate_heloc(heloc_n)
    heloc_path = config.HELOC_RAW_PATH
    df_heloc.to_csv(heloc_path, index=False)
    log.info(f"Saved synthetic HELOC data to {heloc_path}")

    log.info("")
    log.info("Synthetic data generated successfully.")
    log.info("You can now run the Phase 1 notebooks.")
    log.info("Note: Model performance will differ from real-data results.")


if __name__ == "__main__":
    generate_all()
