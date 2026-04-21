"""
notebooks/phase3/03_feature_engineering.py
───────────────────────────────────────────
Phase 3: Feature Engineering, WoE Binning & Reject Inference

Steps:
  1.  Load Phase 1 outputs (train.parquet, test.parquet)
  2.  Apply imputation strategy (from Phase 2 decisions)
  3.  Add interaction features (4 new features)
  4.  Run WoE binning + IV table
  5.  Feature selection (drop IV < 0.02)
  6.  Reject inference (augmentation method)
  7.  PIT → TTC PD transformation setup
  8.  Save final model-ready feature sets
  9.  Phase 3 validation checks

Run as script:   python notebooks/phase3/03_feature_engineering.py
Convert to .ipynb: jupytext --to notebook 03_feature_engineering.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

import config
from src.features.imputation import impute
from src.features.interactions import (
    add_interaction_features,
    apply_reject_inference,
    compute_pit_to_ttc,
    generate_reject_inference,
)
from src.features.woe_binning import (
    run_woe_binning,
    pd_to_score,
    assign_risk_tier,
    BINNING_FEATURES,
    IV_THRESHOLD_DROP,
)

PHASE3_DIR = Path("reports/phase3")
PHASE3_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("CREDIT RISK PLATFORM — PHASE 3: FEATURE ENGINEERING")
print("=" * 65)

# ── Load Phase 1 data ─────────────────────────────────────────────
print("\nLoading Phase 1 outputs ...")
df_train = pd.read_parquet(config.TRAIN_PATH)
df_test  = pd.read_parquet(config.TEST_PATH)
print(f"Train: {df_train.shape}  |  Test: {df_test.shape}")

# %%
# ── STEP 1: IMPUTATION ────────────────────────────────────────────
print("\n" + "─" * 65)
print("Step 1: Imputation")
print("─" * 65)

print("\nMissing values BEFORE imputation:")
nulls_before = df_train.select_dtypes(include="number").isnull().sum()
nulls_before = nulls_before[nulls_before > 0]
for col, n in nulls_before.items():
    pct = n / len(df_train) * 100
    print(f"  {col:45s} {n:6,} ({pct:.1f}%)")

df_train_imp, medians = impute(df_train, fit=True)
df_test_imp,  _       = impute(df_test,  medians=medians, fit=False)

print("\nMissing values AFTER imputation (numeric only):")
nulls_after = df_train_imp.select_dtypes(include="number").isnull().sum()
nulls_after = nulls_after[nulls_after > 0]
if len(nulls_after) == 0:
    print("  All numeric nulls resolved.")
else:
    print(nulls_after.to_string())

# Save medians for production use
joblib.dump(medians, MODELS_DIR / "imputation_medians.pkl")
print(f"\nSaved: models/imputation_medians.pkl")

# %%
# ── STEP 2: INTERACTION FEATURES ─────────────────────────────────
print("\n" + "─" * 65)
print("Step 2: Interaction Features")
print("─" * 65)

df_train_imp = add_interaction_features(df_train_imp)
df_test_imp  = add_interaction_features(df_test_imp)

interaction_cols = ["ltv_x_product", "dti_x_emp_risk",
                    "ads_x_thin_file", "score_x_product"]
print("\nInteraction feature sample (first 6 rows):")
show_cols = ["product_type", "ltv_ratio", "dti",
             "employment_length_years", "alt_data_score",
             "thin_file_flag", "credit_score"] + interaction_cols
print(df_train_imp[show_cols].head(6).to_string(index=False))

print("\nCorrelations of interaction features with default:")
for col in interaction_cols:
    if col in df_train_imp.columns:
        corr = df_train_imp[col].corr(df_train_imp["default_flag"])
        print(f"  {col:30s}  r = {corr:.4f}")

# %%
# ── STEP 3: WoE BINNING ───────────────────────────────────────────
print("\n" + "─" * 65)
print("Step 3: WoE Binning and Information Value")
print("─" * 65)

# Add interaction features to binning list
all_features_to_bin = [
    f for f in BINNING_FEATURES + interaction_cols
    if f in df_train_imp.columns
]

print(f"\nRunning WoE binning on {len(all_features_to_bin)} features ...")
df_train_woe, df_test_woe, iv_table, binning_process = run_woe_binning(
    df_train_imp, df_test_imp,
    features=all_features_to_bin
)

print("\nInformation Value Table:")
print(f"{'Feature':<45} {'IV':>8}  {'Strength'}")
print("─" * 70)
for _, row in iv_table.iterrows():
    flag = "  ← DROP" if row["iv"] < IV_THRESHOLD_DROP else ""
    print(f"  {row['feature']:<43} {row['iv']:>8.4f}  "
          f"{row['strength']}{flag}")

# Save IV table
iv_table.to_csv(PHASE3_DIR / "iv_table.csv", index=False)
print(f"\nSaved: {PHASE3_DIR}/iv_table.csv")

# %%
# ── STEP 4: FEATURE SELECTION ─────────────────────────────────────
print("\n" + "─" * 65)
print("Step 4: Feature Selection (IV threshold)")
print("─" * 65)

keep_features   = iv_table[iv_table["iv"] >= IV_THRESHOLD_DROP]["feature"].tolist()
drop_features   = iv_table[iv_table["iv"] <  IV_THRESHOLD_DROP]["feature"].tolist()
woe_keep_cols   = [f"{f}_woe" for f in keep_features
                   if f"{f}_woe" in df_train_woe.columns]

print(f"\nFeatures KEPT (IV >= {IV_THRESHOLD_DROP}):  {len(keep_features)}")
for f in keep_features:
    iv = iv_table[iv_table["feature"]==f]["iv"].values[0]
    print(f"  {f:<43}  IV = {iv:.4f}")

print(f"\nFeatures DROPPED (IV < {IV_THRESHOLD_DROP}):  {len(drop_features)}")
for f in drop_features:
    iv = iv_table[iv_table["feature"]==f]["iv"].values[0]
    print(f"  {f:<43}  IV = {iv:.4f}")

# Save feature lists
with open(PHASE3_DIR / "selected_features.txt", "w") as f:
    f.write("SELECTED FEATURES (Phase 3)\n")
    f.write("=" * 40 + "\n\n")
    f.write("Numeric features (raw):\n")
    for feat in keep_features:
        iv = iv_table[iv_table["feature"]==feat]["iv"].values[0]
        f.write(f"  {feat:<43}  IV={iv:.4f}\n")

print(f"\nSaved: {PHASE3_DIR}/selected_features.txt")

# %%
# ── STEP 5: REJECT INFERENCE ─────────────────────────────────────
print("\n" + "─" * 65)
print("Step 5: Reject Inference — Augmentation Method")
print("─" * 65)

print("\nStep 5a: Train initial model on approved loans only ...")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Use raw numeric features for initial model
initial_features = [f for f in keep_features
                    if f in df_train_imp.columns
                    and df_train_imp[f].dtype in [np.float64, np.float32,
                                                   np.int64, np.int32, float, int]]

X_train_init = df_train_imp[initial_features].fillna(0)
y_train_init = df_train_imp["default_flag"]

initial_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=300, random_state=42, C=0.1))
])
initial_pipe.fit(X_train_init, y_train_init)

auc_initial = roc_auc_score(
    y_train_init,
    initial_pipe.predict_proba(X_train_init)[:, 1]
)
print(f"  Initial model train AUC: {auc_initial:.4f}")

print("\nStep 5b: Score synthetic declined population ...")
print("Step 5c: Assign probabilistic labels to declined ...")
print("Step 5d: Create augmented training set ...")

df_train_augmented = apply_reject_inference(
    df_train_imp,
    initial_model=initial_pipe,
    feature_cols=initial_features,
    n_declined=5_000,
    bad_rate_assumption=0.55
)

# Train on augmented data
X_train_aug = df_train_augmented[initial_features].fillna(0)
y_train_aug = df_train_augmented["default_flag"]
w_train_aug = df_train_augmented.get("sample_weight",
                                      pd.Series(1.0, index=df_train_augmented.index))

aug_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=300, random_state=42, C=0.1))
])
aug_pipe.fit(X_train_aug, y_train_aug,
             lr__sample_weight=w_train_aug.values)

# Compare AUC on original test set
X_test_eval = df_test_imp[initial_features].fillna(0)
y_test_eval = df_test_imp["default_flag"]

auc_before = roc_auc_score(
    y_test_eval,
    initial_pipe.predict_proba(X_test_eval)[:, 1]
)
auc_after = roc_auc_score(
    y_test_eval,
    aug_pipe.predict_proba(X_test_eval)[:, 1]
)

print(f"\nReject Inference Impact:")
print(f"  AUC before reject inference: {auc_before:.4f}")
print(f"  AUC after reject inference:  {auc_after:.4f}")
print(f"  AUC lift:                    {auc_after - auc_before:+.4f}")
print(f"  Note: Lift will be larger on real data in low-score band (550-620)")
print(f"  Synthetic data has no real rejected population to recover")

# Save augmented training data
aug_path = config.PROCESSED_DIR / "train_augmented.parquet"
df_train_augmented.to_parquet(aug_path, index=False)
print(f"\nSaved: {aug_path}")

# %%
# ── STEP 6: PIT vs TTC PD ────────────────────────────────────────
print("\n" + "─" * 65)
print("Step 6: PIT vs TTC PD Transformation")
print("─" * 65)

print("""
EXPLANATION:
  PIT (Point-in-Time) PD: What the model predicts RIGHT NOW
    → Used for: IFRS 9 ECL provisioning
    → Volatile: rises in recessions, falls in booms
    → This is what XGBoost produces

  TTC (Through-the-Cycle) PD: Long-run average, cycle-smoothed
    → Used for: Basel III regulatory capital (RWA)
    → Stable: changes slowly, anchored to long-run average
    → Derived from PIT via smoothing toward long-run mean
""")

# Demonstrate on a sample of PD scores
sample_pd_pit = np.array([0.05, 0.12, 0.20, 0.35, 0.50, 0.65, 0.80])
sample_pd_ttc = compute_pit_to_ttc(
    sample_pd_pit,
    long_run_average_pd=0.08,
    smoothing_factor=0.30
)

print("PIT vs TTC comparison (long-run avg PD = 8%, smoothing = 0.30):")
print(f"  {'PIT PD':>10}  {'TTC PD':>10}  {'Difference':>12}  Notes")
print("  " + "─" * 55)
for pit, ttc in zip(sample_pd_pit, sample_pd_ttc):
    diff = ttc - pit
    note = ""
    if pit < 0.08:
        note = "TTC pulls UP  (boom period)"
    elif pit > 0.08:
        note = "TTC pulls DOWN (recession period)"
    else:
        note = "No adjustment (at long-run avg)"
    print(f"  {pit:>10.2%}  {ttc:>10.2%}  {diff:>+12.2%}  {note}")

print("""
INTERPRETATION:
  In a boom (low PD environment): TTC > PIT
    → Forces lenders to hold MORE capital than current risk suggests
    → Prevents capital release during good times
  
  In a recession (high PD environment): TTC < PIT
    → Dampens spike in required capital
    → Prevents procyclical capital reduction
  
  This is exactly why Basel III requires TTC, not PIT, for Pillar 1.
""")

# %%
# ── STEP 7: SAVE FINAL FEATURE SETS ──────────────────────────────
print("\n" + "─" * 65)
print("Step 7: Save Final Model-Ready Feature Sets")
print("─" * 65)

# Define final feature set for modelling
# Both raw features AND WoE-transformed versions
meta_cols = ["product_type", "data_source", "origination_year",
             "origination_quarter", "default_flag",
             "thin_file_flag", "sample_weight"]

# Raw features for XGBoost/tree models
raw_model_features = [f for f in keep_features
                       if f in df_train_imp.columns]

# WoE features for logistic regression scorecard
woe_model_features = [f"{f}_woe" for f in keep_features
                       if f"{f}_woe" in df_train_woe.columns]

print(f"\nRaw features for tree models: {len(raw_model_features)}")
for f in raw_model_features:
    print(f"  {f}")

print(f"\nWoE features for scorecard: {len(woe_model_features)}")
for f in woe_model_features:
    print(f"  {f}")

# Save train (with WoE columns)
train_final_path = config.PROCESSED_DIR / "train_features.parquet"
test_final_path  = config.PROCESSED_DIR / "test_features.parquet"
aug_final_path   = config.PROCESSED_DIR / "train_augmented_features.parquet"

df_train_woe.to_parquet(train_final_path, index=False)
df_test_woe.to_parquet(test_final_path, index=False)

# Add WoE cols to augmented set
df_aug_woe = df_train_augmented.copy()
for col in woe_model_features:
    raw_col = col.replace("_woe", "")
    if raw_col in df_aug_woe.columns:
        if col in df_train_woe.columns:
            df_aug_woe[col] = np.nan
df_aug_woe.to_parquet(aug_final_path, index=False)

# Save feature lists as artifacts
feature_config = {
    "raw_features":  raw_model_features,
    "woe_features":  woe_model_features,
    "keep_features": keep_features,
    "drop_features": drop_features,
    "interaction_features": interaction_cols,
    "target": "default_flag",
}
joblib.dump(feature_config, MODELS_DIR / "feature_config.pkl")
if binning_process is not None:
    joblib.dump(binning_process, config.BINNING_PATH)
    print(f"\nSaved: {config.BINNING_PATH}")

print(f"\nSaved: {train_final_path}")
print(f"Saved: {test_final_path}")
print(f"Saved: {aug_final_path}")
print(f"Saved: models/feature_config.pkl")

# %%
# ── STEP 8: VALIDATION ────────────────────────────────────────────
print("\n" + "─" * 65)
print("Step 8: Phase 3 Validation Checks")
print("─" * 65)

def chk(name, condition, val=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name} {val}")
    return condition

all_pass = True
print()
# Exclude metadata columns from null check
meta_exclude = ["has_synthetic_features", "origination_year"]
model_numeric = [c for c in df_train_woe.select_dtypes(include="number").columns
                 if c not in meta_exclude]
all_pass &= chk("No numeric nulls in model features (train)",
                df_train_woe[model_numeric].isnull().sum().sum() == 0)
all_pass &= chk("No numeric nulls in model features (test)",
                df_test_woe[model_numeric].isnull().sum().sum() == 0)
all_pass &= chk("Interaction features added",
                all(c in df_train_woe.columns for c in interaction_cols))
all_pass &= chk("IV table has entries",
                len(iv_table) > 0,
                f"({len(iv_table)} features evaluated)")
all_pass &= chk("At least 5 features kept after IV filter",
                len(keep_features) >= 5,
                f"({len(keep_features)} kept)")
all_pass &= chk("Augmented training set larger than original",
                len(df_train_augmented) > len(df_train),
                f"({len(df_train_augmented):,} vs {len(df_train):,})")
all_pass &= chk("AUC after reject inference within tolerance",
                auc_after >= auc_before - 0.02,
                f"({auc_after:.4f} vs {auc_before:.4f})")
all_pass &= chk("Feature config saved",
                (MODELS_DIR / "feature_config.pkl").exists())
all_pass &= chk("Train features file saved",
                train_final_path.exists())
all_pass &= chk("Test features file saved",
                test_final_path.exists())

print()
if all_pass:
    print("  ALL CHECKS PASSED — Phase 3 complete.")
else:
    print("  SOME CHECKS FAILED — review output above.")

# %%
# ── STEP 9: PHASE 3 SUMMARY ──────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 3 COMPLETE")
print("=" * 65)

print(f"""
Summary:
  Original train rows:      {len(df_train):,}
  Augmented train rows:     {len(df_train_augmented):,}
  Test rows:                {len(df_test):,}
  
  Features before IV filter: {len(all_features_to_bin)}
  Features after IV filter:  {len(keep_features)} raw + {len(woe_model_features)} WoE
  Features dropped:          {len(drop_features)}
  
  Reject inference AUC lift: {auc_after - auc_before:+.4f}
  
  Output files:
    data/processed/train_features.parquet
    data/processed/test_features.parquet
    data/processed/train_augmented.parquet
    data/processed/train_augmented_features.parquet
    models/feature_config.pkl
    models/imputation_medians.pkl
    reports/phase3/iv_table.csv
    reports/phase3/selected_features.txt

Next: Phase 4 — Model Training (PD + LGD + EAD)
  Run: notebooks/phase4/04_model_training.py
""")
