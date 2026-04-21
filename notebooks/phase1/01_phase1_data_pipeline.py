# %%
"""
notebooks/phase1/01_phase1_data_pipeline.py
────────────────────────────────────────────
Phase 1: Data Acquisition, Cleaning & Harmonization

This notebook runs the complete Phase 1 pipeline:
  1. LendingClub data acquisition and cleaning
  2. FICO HELOC data acquisition and cleaning
  3. Dataset harmonization (unified schema + product_type flag)
  4. Alternative data composite score engineering
  5. Out-of-time train/test split
  6. Phase 1 validation report

Convert to Jupyter notebook with:
  jupytext --to notebook 01_phase1_data_pipeline.py

Run as a script with:
  python notebooks/phase1/01_phase1_data_pipeline.py
"""

# ── Imports ──────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import config
from src.data.sample_generator import generate_all
from src.data.lendingclub import load_and_process as process_lc
from src.data.heloc import load_and_process as process_heloc
from src.data.harmonize import run_harmonization
from src.data.alt_data import add_alt_data_features, generate_ads_methodology_doc

print("=" * 65)
print("CREDIT RISK PLATFORM — PHASE 1: DATA PIPELINE")
print("=" * 65)

# %%
# ── Step 0: Generate synthetic data if real data not present ─────
print("\nStep 0: Checking for data files ...")

lc_exists    = config.LC_RAW_PATH.exists()
heloc_exists = config.HELOC_RAW_PATH.exists()

if not lc_exists or not heloc_exists:
    print("Real datasets not found. Generating synthetic sample data ...")
    print("(For real results, download datasets per data/README.md)")
    generate_all()
    print("Synthetic data generated.\n")
else:
    print(f"Found: {config.LC_RAW_PATH}")
    print(f"Found: {config.HELOC_RAW_PATH}")

# %%
# ── Step 1: Process LendingClub ───────────────────────────────────
print("\n" + "─" * 65)
print("Step 1: LendingClub Data Processing")
print("─" * 65)

df_lc = process_lc(save=True)

print(f"\nLendingClub Summary:")
print(f"  Shape:              {df_lc.shape}")
print(f"  Default rate:       {df_lc['default_flag'].mean():.2%}")
print(f"  Date range:         {df_lc['origination_year'].min():.0f}–"
      f"{df_lc['origination_year'].max():.0f}")
print(f"  Avg loan amount:    ${df_lc['loan_amount'].mean():,.0f}")
print(f"  Avg credit score:   {df_lc['credit_score'].mean():.0f}")
print(f"  Avg DTI:            {df_lc['dti'].mean():.1f}%")
print(f"  Null rate (dti):    {df_lc['dti'].isna().mean():.2%}")

# Default rate by vintage
print("\n  Default rate by origination year:")
vintage_dr = (df_lc.groupby("origination_year")["default_flag"]
              .agg(["count", "mean"])
              .rename(columns={"count": "n_loans", "mean": "default_rate"}))
vintage_dr["default_rate"] = vintage_dr["default_rate"].map("{:.2%}".format)
print(vintage_dr.to_string())

# %%
# ── Step 2: Process HELOC ─────────────────────────────────────────
print("\n" + "─" * 65)
print("Step 2: FICO HELOC Data Processing")
print("─" * 65)

df_heloc = process_heloc(save=True)

print(f"\nHELOC Summary:")
print(f"  Shape:              {df_heloc.shape}")
print(f"  Default rate:       {df_heloc['default_flag'].mean():.2%}")
print(f"  Avg credit score:   {df_heloc['credit_score'].mean():.0f}")
print(f"  Avg LTV ratio:      {df_heloc['ltv_ratio'].mean():.2%}")
print(f"  Avg DTI (proxy):    {df_heloc['dti'].mean():.1f}%")
print(f"  Note: HELOC loan_amount, annual_income, dti,")
print(f"        ltv_ratio are synthetic proxies (see src/data/heloc.py)")

# %%
# ── Step 3: Harmonize datasets ────────────────────────────────────
print("\n" + "─" * 65)
print("Step 3: Dataset Harmonization")
print("─" * 65)

df_harmonized, df_train, df_test = run_harmonization(save=True)

print(f"\nHarmonized Dataset Summary:")
print(f"  Total rows:          {len(df_harmonized):,}")
print(f"  Columns:             {df_harmonized.shape[1]}")
print(f"  Unsecured (LC):      {(df_harmonized['product_type']==0).sum():,}")
print(f"  Secured (HELOC):     {(df_harmonized['product_type']==1).sum():,}")
print(f"  Overall default:     {df_harmonized['default_flag'].mean():.2%}")
print(f"\nTrain/Test Split (out-of-time):")
print(f"  Train: {len(df_train):,} rows | default: {df_train['default_flag'].mean():.2%}")
print(f"  Test:  {len(df_test):,} rows  | default: {df_test['default_flag'].mean():.2%}")

# %%
# ── Step 4: Alternative data score ───────────────────────────────
print("\n" + "─" * 65)
print("Step 4: Alternative Data Composite Score Engineering")
print("─" * 65)

# Add to harmonized dataset
df_harmonized = add_alt_data_features(df_harmonized)
df_train      = add_alt_data_features(df_train)
df_test       = add_alt_data_features(df_test)

# ADS lift analysis by thin-file status
print("\nADS Lift Analysis:")
for product, pname in [(0, "Unsecured"), (1, "Secured")]:
    subset = df_harmonized[df_harmonized["product_type"] == product]
    for thin, tname in [(True, "Thin-file"), (False, "Standard")]:
        group = subset[subset["thin_file_flag"] == thin]
        if len(group) > 0:
            print(f"  {pname} / {tname}: "
                  f"n={len(group):,}  "
                  f"default={group['default_flag'].mean():.2%}  "
                  f"avg_ADS={group['alt_data_score'].mean():.1f}")

# Correlation of ADS with default by segment
print("\nADS-Default Correlation by Product Type:")
for product, pname in [(0, "Unsecured"), (1, "Secured")]:
    subset = df_harmonized[df_harmonized["product_type"] == product]
    corr = subset["alt_data_score"].corr(subset["default_flag"])
    print(f"  {pname}: r = {corr:.3f}")

# Save updated versions with alt data
df_harmonized.to_parquet(config.HARMONIZED_PATH, index=False)
df_train.to_parquet(config.TRAIN_PATH, index=False)
df_test.to_parquet(config.TEST_PATH, index=False)
print("\nSaved updated datasets with alt_data_score and thin_file_flag.")

# %%
# ── Step 5: Phase 1 Validation Report ────────────────────────────
print("\n" + "─" * 65)
print("Step 5: Phase 1 Validation Report")
print("─" * 65)

def check(name, condition, value=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name} {value}")
    return condition

all_pass = True
print("\nData Quality Checks:")
all_pass &= check("Harmonized dataset rows > 50K",
                  len(df_harmonized) > 50_000,
                  f"({len(df_harmonized):,})")
all_pass &= check("product_type in {0,1}",
                  df_harmonized["product_type"].isin([0, 1]).all())
all_pass &= check("default_flag in {0,1}",
                  df_harmonized["default_flag"].isin([0, 1]).all())
all_pass &= check("Overall default rate 10-50%",
                  0.10 <= df_harmonized["default_flag"].mean() <= 0.50,
                  f"({df_harmonized['default_flag'].mean():.2%})")
all_pass &= check("Train size > test size",
                  len(df_train) > len(df_test))
all_pass &= check("alt_data_score range [0,100]",
                  df_harmonized["alt_data_score"].between(0, 100).all())
all_pass &= check("thin_file_flag is boolean",
                  df_harmonized["thin_file_flag"].dtype == bool)
all_pass &= check("ADS negatively correlated with default",
                  df_harmonized["alt_data_score"].corr(
                      df_harmonized["default_flag"]) < 0)
all_pass &= check("Train data saved",
                  config.TRAIN_PATH.exists())
all_pass &= check("Test data saved",
                  config.TEST_PATH.exists())
all_pass &= check("Harmonized data saved",
                  config.HARMONIZED_PATH.exists())

print()
if all_pass:
    print("  ALL CHECKS PASSED — Phase 1 complete.")
else:
    print("  SOME CHECKS FAILED — review output above.")

# %%
# ── Step 6: Feature completeness matrix ──────────────────────────
print("\n" + "─" * 65)
print("Step 6: Feature Completeness Matrix")
print("─" * 65)

from src.data.harmonize import NUMERIC_FEATURES
key_features = [f for f in NUMERIC_FEATURES if f in df_harmonized.columns]

completeness = pd.DataFrame({
    "feature": key_features,
    "unsecured_null_%": [
        df_harmonized[df_harmonized["product_type"]==0][f].isna().mean() * 100
        for f in key_features
    ],
    "secured_null_%": [
        df_harmonized[df_harmonized["product_type"]==1][f].isna().mean() * 100
        for f in key_features
    ],
    "source": [
        "N" if f in ["dti", "credit_score", "loan_amount", "annual_income",
                     "total_accounts", "num_inquiries_last_6m",
                     "credit_utilization", "num_derogatory_marks"]
        else "S" if f in ["ltv_ratio", "employment_length_years"]
        else "E"   # Engineered (alt data, product type)
        for f in key_features
    ]
})
completeness["unsecured_null_%"] = completeness["unsecured_null_%"].round(1)
completeness["secured_null_%"]   = completeness["secured_null_%"].round(1)

print("\nKey: N=Native  S=Synthetic proxy  E=Engineered")
print(completeness.to_string(index=False))

# %%
# ── Step 7: ADS Methodology Documentation ────────────────────────
print("\n" + "─" * 65)
print("Step 7: Alternative Data Methodology")
print("─" * 65)
print(generate_ads_methodology_doc())

# %%
print("\n" + "=" * 65)
print("PHASE 1 COMPLETE")
print("=" * 65)
print(f"\nOutput files:")
print(f"  {config.HARMONIZED_PATH}")
print(f"  {config.TRAIN_PATH}")
print(f"  {config.TEST_PATH}")
print(f"\nNext: Phase 2 — Exploratory Data Analysis")
print(f"  Run: notebooks/phase2/02_eda.py")
print("=" * 65)
