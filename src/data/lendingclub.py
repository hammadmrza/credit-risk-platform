"""
src/data/lendingclub.py
───────────────────────
LendingClub data acquisition and cleaning module.

Handles:
- Chunked loading of the large CSV (avoids OOM on ~1.5GB file)
- Filtering to settled loans only (Fully Paid / Charged Off)
- Stratified sampling to 200K records
- Feature selection and type casting
- Target variable encoding
- Origination date parsing for vintage analysis
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


# ── Feature selection ────────────────────────────────────────────────────────
# These are the raw LendingClub column names we retain.
# We select only features that are available at application time (no leakage).
LC_KEEP_COLS = [
    # Loan characteristics
    "loan_amnt",          # Loan amount requested
    "term",               # 36 or 60 months
    "int_rate",           # Interest rate
    "installment",        # Monthly payment
    "grade",              # LC internal grade (A-G)
    "sub_grade",          # LC sub-grade
    "purpose",            # Loan purpose
    # Borrower characteristics
    "emp_length",         # Employment length
    "home_ownership",     # RENT / OWN / MORTGAGE
    "annual_inc",         # Annual income (stated)
    "verification_status",# Income verification status
    "addr_state",         # State
    # Bureau features
    "dti",                # Debt-to-income ratio
    "delinq_2yrs",        # Delinquencies in past 2 years
    "inq_last_6mths",     # Credit inquiries last 6 months
    "mths_since_last_delinq",  # Months since last delinquency
    "open_acc",           # Number of open credit lines
    "pub_rec",            # Public derogatory records
    "revol_bal",          # Revolving balance
    "revol_util",         # Revolving utilisation rate
    "total_acc",          # Total credit accounts
    "mort_acc",           # Number of mortgage accounts
    "pub_rec_bankruptcies",  # Public record bankruptcies
    "fico_range_low",     # FICO range lower bound
    "fico_range_high",    # FICO range upper bound
    "earliest_cr_line",   # Date of earliest credit line
    # Target and time
    "loan_status",        # Used to derive default flag
    "issue_d",            # Origination date for vintage analysis
]

# Columns where negative values are impossible and should be treated as null
NON_NEGATIVE_COLS = [
    "dti", "annual_inc", "revol_util", "loan_amnt",
    "inq_last_6mths", "delinq_2yrs", "open_acc", "pub_rec",
    "fico_range_low", "fico_range_high",
]


def load_raw(path: Optional[Path] = None,
             chunksize: int = 50_000) -> pd.DataFrame:
    """
    Load LendingClub CSV using chunked reading to manage memory.

    Args:
        path: Path to the raw CSV. Defaults to config.LC_RAW_PATH.
        chunksize: Rows per chunk. 50K is a safe default.

    Returns:
        DataFrame containing only settled loans with kept columns.
    """
    path = path or config.LC_RAW_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"LendingClub raw file not found at {path}\n"
            "See data/README.md for download instructions.\n"
            "Or run: python src/data/sample_generator.py"
        )

    log.info(f"Loading LendingClub data from {path} ...")
    log.info("Using chunked reader — this may take 1-2 minutes for the full file.")

    chunks = []
    total_rows = 0
    settled_rows = 0

    # Identify which columns exist in this version of the file
    # (LendingClub changed schema over the years)
    header = pd.read_csv(path, nrows=0)
    available_cols = [c for c in LC_KEEP_COLS if c in header.columns]
    missing_cols = [c for c in LC_KEEP_COLS if c not in header.columns]

    if missing_cols:
        log.warning(f"These expected columns not found in file: {missing_cols}")
        log.warning("Processing will continue with available columns.")

    for i, chunk in enumerate(pd.read_csv(
            path,
            usecols=available_cols,
            low_memory=False,
            chunksize=chunksize)):

        total_rows += len(chunk)

        # Filter to settled loans immediately to reduce memory
        mask = chunk["loan_status"].isin(config.LC_TARGET_STATUSES)
        chunk = chunk[mask].copy()
        settled_rows += len(chunk)
        chunks.append(chunk)

        if (i + 1) % 10 == 0:
            log.info(f"  Processed {total_rows:,} rows, "
                     f"{settled_rows:,} settled loans retained ...")

    df = pd.concat(chunks, ignore_index=True)
    log.info(f"Loaded {total_rows:,} total rows → "
             f"{settled_rows:,} settled loans ({100*settled_rows/total_rows:.1f}%)")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning rules to the raw LendingClub DataFrame.

    Cleaning steps:
    1. Encode target variable
    2. Parse dates
    3. Clean string columns
    4. Handle missing values
    5. Feature engineering (credit_score, employment_length_years)
    6. Remove impossible values

    Args:
        df: Raw LendingClub DataFrame (output of load_raw).

    Returns:
        Cleaned DataFrame with standardised column names.
    """
    log.info("Cleaning LendingClub data ...")
    df = df.copy()

    # ── 1. Target variable ────────────────────────────────────────
    df["default_flag"] = (
        df["loan_status"].isin(config.LC_BAD_STATUSES)
    ).astype(int)

    # ── 2. Parse origination date ─────────────────────────────────
    df["origination_date"] = pd.to_datetime(
        df["issue_d"], format="%b-%Y", errors="coerce"
    )
    df["origination_year"]  = df["origination_date"].dt.year
    df["origination_quarter"] = df["origination_date"].dt.to_period("Q").astype(str)

    # ── 3. Clean string columns ───────────────────────────────────
    # Interest rate: "13.56%" → 13.56
    if "int_rate" in df.columns:
        df["int_rate"] = (
            df["int_rate"].astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
        )

    # Revolving utilisation: "54.3%" → 54.3
    if "revol_util" in df.columns:
        df["revol_util"] = (
            df["revol_util"].astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
        )

    # Loan term: " 36 months" → 36
    if "term" in df.columns:
        df["loan_term_months"] = (
            df["term"].astype(str)
            .str.extract(r"(\d+)")[0]
            .pipe(pd.to_numeric, errors="coerce")
        )

    # Employment length: "10+ years" → 10, "< 1 year" → 0
    if "emp_length" in df.columns:
        emp_map = {
            "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
            "4 years": 4,  "5 years": 5, "6 years": 6, "7 years": 7,
            "8 years": 8,  "9 years": 9, "10+ years": 10
        }
        df["employment_length_years"] = (
            df["emp_length"].map(emp_map)
        )

    # ── 4. Credit score (midpoint of FICO range) ──────────────────
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["credit_score"] = (
            df["fico_range_low"].fillna(0) + df["fico_range_high"].fillna(0)
        ) / 2
        df.loc[df["credit_score"] == 0, "credit_score"] = np.nan

    # ── 5. Months since oldest trade ─────────────────────────────
    if "earliest_cr_line" in df.columns:
        earliest = pd.to_datetime(
            df["earliest_cr_line"], format="%b-%Y", errors="coerce"
        )
        df["months_since_oldest_trade"] = (
            (df["origination_date"] - earliest)
            .dt.days / 30.44
        ).round().astype("Int64")

    # ── 6. Remove impossible / extreme values ─────────────────────
    for col in NON_NEGATIVE_COLS:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Cap DTI at 100 (some LC records have >100% which are errors)
    if "dti" in df.columns:
        df.loc[df["dti"] > 100, "dti"] = np.nan

    # Cap annual income at $500K (obvious outlier / error above that)
    if "annual_inc" in df.columns:
        df.loc[df["annual_inc"] > 500_000, "annual_inc"] = np.nan

    # ── 7. Rename to harmonized schema ───────────────────────────
    rename_map = {
        "loan_amnt":       "loan_amount",
        "annual_inc":      "annual_income",
        "dti":             "dti",
        "revol_util":      "credit_utilization",
        "open_acc":        "num_open_accounts",
        "pub_rec":         "num_derogatory_marks",
        "total_acc":       "total_accounts",
        "inq_last_6mths":  "num_inquiries_last_6m",
        "delinq_2yrs":     "num_delinquencies_2yr",
        "mths_since_last_delinq": "months_since_recent_delinquency",
        "mort_acc":        "num_mortgage_accounts",
        "pub_rec_bankruptcies": "num_bankruptcies",
        "revol_bal":       "revolving_balance",
        "installment":     "monthly_payment",
        "int_rate":        "interest_rate",
        "grade":           "lc_grade",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items()
                             if k in df.columns})

    # ── 8. Add product type flag ──────────────────────────────────
    df["product_type"] = config.PRODUCT_TYPE_UNSECURED
    df["ltv_ratio"]    = np.nan     # Not applicable for unsecured

    log.info(f"Cleaning complete. Shape: {df.shape}")
    log.info(f"Default rate: {df['default_flag'].mean():.2%}")
    return df


def sample_stratified(df: pd.DataFrame,
                       n: int = config.LC_SAMPLE_SIZE,
                       random_state: int = config.LC_RANDOM_STATE
                       ) -> pd.DataFrame:
    """
    Stratified sample maintaining the bad rate.

    Args:
        df: Cleaned LendingClub DataFrame.
        n: Target sample size.
        random_state: Reproducibility seed.

    Returns:
        Stratified sample of n rows.
    """
    if len(df) <= n:
        log.info(f"Dataset ({len(df):,}) smaller than target ({n:,}). "
                 "Returning full dataset.")
        return df

    log.info(f"Stratified sampling {n:,} records from {len(df):,} ...")

    from sklearn.model_selection import train_test_split
    _, sample = train_test_split(
        df,
        test_size=n / len(df),
        stratify=df["default_flag"],
        random_state=random_state
    )

    log.info(f"Sample shape: {sample.shape}")
    log.info(f"Sample default rate: {sample['default_flag'].mean():.2%}")
    return sample.reset_index(drop=True)


def load_and_process(path: Optional[Path] = None,
                     save: bool = True) -> pd.DataFrame:
    """
    Full pipeline: load → clean → sample → save.

    Args:
        path: Path to raw CSV. Defaults to config.LC_RAW_PATH.
        save: Whether to save the processed file to config.LC_CLEAN_PATH.

    Returns:
        Cleaned, sampled LendingClub DataFrame.
    """
    df_raw     = load_raw(path)
    df_clean   = clean(df_raw)
    df_sample  = sample_stratified(df_clean)

    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_sample.to_parquet(config.LC_CLEAN_PATH, index=False)
        log.info(f"Saved to {config.LC_CLEAN_PATH}")

    return df_sample


if __name__ == "__main__":
    df = load_and_process()
    print(df.head())
    print(df.dtypes)
