"""
notebooks/phase4b/04b_segmented_models.py
──────────────────────────────────────────
Phase 4b — Separate product models (v1.1).

Trains per-product PD models (Unsecured + Secured) alongside the
existing v1.0 unified model. Produces a comparison table showing
the AUC/KS/Gini lift from segmentation.

This script is a SIDEBAR to the main pipeline — it does NOT replace
Phase 4. It demonstrates the v1.1 segmented approach for the
portfolio's model performance tab.

Run after Phase 1-4 have produced the processed training / OOT parquet files.

USAGE
─────
    python notebooks/phase4b/04b_segmented_models.py

OUTPUTS
───────
    models/xgb_pd_unsecured.pkl
    models/xgb_pd_secured.pkl
    models/xgb_pd_unsecured_features.pkl
    models/xgb_pd_secured_features.pkl
    reports/phase4/model_comparison_segmented.csv
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pd_model_segmented import (
    train_segmented_models,
    results_to_dataframe,
    save_artifacts,
)


# ── Expected data paths — written by Phase 1 + 3 ──────────────────
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_woe.parquet"
TEST_PATH  = PROJECT_ROOT / "data" / "processed" / "test_oot_woe.parquet"
# Fallbacks if the WoE-transformed files aren't available — use raw harmonized
TRAIN_FALLBACK = PROJECT_ROOT / "data" / "processed" / "train_harmonized.parquet"
TEST_FALLBACK  = PROJECT_ROOT / "data" / "processed" / "test_oot_harmonized.parquet"

TARGET = "default_flag"


def load_train_test() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load train + OOT test data.
    Prefers WoE-transformed; falls back to raw harmonized.
    """
    if TRAIN_PATH.exists() and TEST_PATH.exists():
        tr = pd.read_parquet(TRAIN_PATH)
        te = pd.read_parquet(TEST_PATH)
        print(f"[OK] Loaded WoE-transformed data: train={len(tr):,}, test={len(te):,}")
    elif TRAIN_FALLBACK.exists() and TEST_FALLBACK.exists():
        tr = pd.read_parquet(TRAIN_FALLBACK)
        te = pd.read_parquet(TEST_FALLBACK)
        print(f"[OK] Loaded raw harmonized data: train={len(tr):,}, test={len(te):,}")
    else:
        raise FileNotFoundError(
            "No processed data found. Run Phase 1-3 first:\n"
            "  python notebooks/phase1/01_phase1_data_pipeline.py\n"
            "  python notebooks/phase3/03_feature_engineering.py"
        )

    if TARGET not in tr.columns:
        raise ValueError(f"Target column '{TARGET}' missing from training data")

    y_train = tr[TARGET].astype(int)
    y_test  = te[TARGET].astype(int)

    # Drop target + metadata columns from feature set
    drop_cols = {TARGET, "origination_year", "data_source",
                 "has_synthetic_features", "issue_d", "earliest_cr_line"}
    X_train = tr.drop(columns=[c for c in drop_cols if c in tr.columns])
    X_test  = te.drop(columns=[c for c in drop_cols if c in te.columns])

    return X_train, y_train, X_test, y_test


def main() -> None:
    print("=" * 65)
    print("Phase 4b — Segmented PD Models (v1.1)")
    print("=" * 65)

    X_train, y_train, X_test, y_test = load_train_test()

    print(f"\nProduct distribution (train):")
    print(f"  Unsecured (0): {(X_train['product_type']==0).sum():>8,}")
    print(f"  Secured (1):   {(X_train['product_type']==1).sum():>8,}")
    print(f"\nProduct distribution (OOT test):")
    print(f"  Unsecured (0): {(X_test ['product_type']==0).sum():>8,}")
    print(f"  Secured (1):   {(X_test ['product_type']==1).sum():>8,}")

    print("\n" + "─" * 65)
    print("Training per-product models ...")
    print("─" * 65)

    results = train_segmented_models(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 65)
    print("RESULTS — Segmented Model Performance")
    print("=" * 65)
    df = results_to_dataframe(results)
    if len(df) > 0:
        print(df.to_string(index=False))

    # Compare against v1.0 unified model
    unified_path = PROJECT_ROOT / "reports" / "phase4" / "model_comparison.csv"
    if unified_path.exists():
        print("\n" + "─" * 65)
        print("COMPARISON vs v1.0 Unified Model")
        print("─" * 65)
        unified = pd.read_csv(unified_path)
        print("\nv1.0 Unified (combined portfolio):")
        print(unified.to_string(index=False))

        # Summary numbers for the tab 3 narrative
        unif_test_auc = unified[unified["model"].str.contains("XGBoost")]["auc"].iloc[0]
        seg_test = df[df["split"] == "Test (OOT)"]
        if len(seg_test) > 0:
            uns_auc = seg_test[seg_test["product"] == "Unsecured"]["auc"]
            sec_auc = seg_test[seg_test["product"] == "Secured"]["auc"]
            print("\nLIFT SUMMARY")
            print(f"  v1.0 Unified OOT AUC:        {unif_test_auc:.4f}")
            if len(uns_auc) > 0:
                print(f"  v1.1 Unsecured-only OOT AUC: {float(uns_auc.iloc[0]):.4f}  "
                      f"(Δ +{float(uns_auc.iloc[0]) - unif_test_auc:.4f})")
            if len(sec_auc) > 0:
                print(f"  v1.1 Secured-only OOT AUC:   {float(sec_auc.iloc[0]):.4f}  "
                      f"(Δ +{float(sec_auc.iloc[0]) - unif_test_auc:.4f})")

    save_artifacts(results,
                   models_dir=PROJECT_ROOT / "models",
                   reports_dir=PROJECT_ROOT / "reports" / "phase4")

    print("\n" + "=" * 65)
    print("Phase 4b COMPLETE")
    print("=" * 65)
    print("""
Artifacts written:
  models/xgb_pd_unsecured.pkl
  models/xgb_pd_secured.pkl
  reports/phase4/model_comparison_segmented.csv

Tab 3 of the Streamlit app will automatically display the segmented
comparison when this CSV is present.
""")


if __name__ == "__main__":
    main()
