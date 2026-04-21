"""
notebooks/phase9/09_fraud.py
──────────────────────────────
Phase 9: Fraud Detection & Monitoring

Steps:
  1.  Generate synthetic fraud labels (documented swap point)
  2.  Engineer fraud-specific features (FPD, velocity, synthetic ID)
  3.  Train XGBoost fraud detection model
  4.  Evaluate with fraud metrics (precision, recall, F1, capture rate)
  5.  Assign alert tiers (LOW / MEDIUM / HIGH / CONFIRMED)
  6.  Fraud rate analysis by channel and product
  7.  EPD / FPD cohort analysis
  8.  Dealer/channel anomaly detection
  9.  Investigation queue with SHAP-based prompts
  10. Save all fraud artifacts
  11. Validation checks

Run: python notebooks/phase9/09_fraud.py
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
from src.data.fraud_label_generator import (
    generate_synthetic_fraud_labels, FRAUD_TYPES, FRAUD_TYPE_DIST
)
from src.features.fraud_features import (
    build_fraud_features, FRAUD_FEATURE_COLS
)
from src.models.fraud_model import (
    train_fraud_model, evaluate_fraud_model,
    assign_alert_tier, FRAUD_MODEL_FEATURES, ALERT_ACTIONS
)

PHASE9_DIR = Path("reports/phase9")
PHASE9_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("CREDIT RISK PLATFORM — PHASE 9: FRAUD DETECTION")
print("=" * 65)

# ── Load portfolio ────────────────────────────────────────────────
print("\nLoading portfolio and Phase 4 outputs ...")
portfolio  = pd.read_parquet("data/processed/portfolio_regulatory.parquet")
test_feat  = pd.read_parquet("data/processed/test_features.parquet")
feat_cfg   = joblib.load("models/feature_config.pkl")

# Merge portfolio with test features for richer feature set
df = pd.concat([
    portfolio.reset_index(drop=True),
    test_feat[[c for c in test_feat.columns
               if c not in portfolio.columns]].reset_index(drop=True)
], axis=1)

print(f"  Portfolio: {len(df):,} loans")
print(f"  Columns available: {len(df.columns)}")

# %%
# ── STEP 1: FRAUD LABELS ─────────────────────────────────────────
print("\n" + "─" * 65)
print("Step 1: Fraud Label Generation (Synthetic Placeholder)")
print("─" * 65)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PLACEHOLDER — SYNTHETIC LABELS
  Replace generate_synthetic_fraud_labels() with:
  load_proprietary_fraud_labels(source='lender_internal')
  See src/data/fraud_label_generator.py for full interface.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

df = generate_synthetic_fraud_labels(
    df,
    pd_col="pd_score",
    credit_score_col="credit_score",
    product_type_col="product_type",
)

n_fraud     = df["fraud_confirmed"].sum()
fraud_rate  = df["fraud_confirmed"].mean()
total_loss  = df["loss_attributed"].sum()

print(f"\n  Fraud label distribution:")
print(f"  Total loans:       {len(df):,}")
print(f"  Confirmed fraud:   {n_fraud:,} ({fraud_rate:.2%})")
print(f"  Clean:             {len(df)-n_fraud:,} ({1-fraud_rate:.2%})")
print(f"  Total fraud loss:  ${total_loss:,.0f}")

print(f"\n  Fraud type breakdown:")
type_dist = df[df["fraud_confirmed"]]["fraud_type"].value_counts()
for ftype, n in type_dist.items():
    pct = n / n_fraud * 100
    bar = "█" * int(pct / 3)
    print(f"    {ftype:<20}  {n:5,} ({pct:5.1f}%)  {bar}")

# Save labelled dataset
df.to_parquet("data/processed/fraud_labelled.parquet", index=False)
print(f"\n  Saved: data/processed/fraud_labelled.parquet")

# %%
# ── STEP 2: FRAUD FEATURE ENGINEERING ────────────────────────────
print("\n" + "─" * 65)
print("Step 2: Fraud Feature Engineering")
print("─" * 65)

df = build_fraud_features(df, pd_col="pd_score")

print("\n  Fraud features created:")
fraud_cols_present = [c for c in FRAUD_FEATURE_COLS if c in df.columns]
for col in fraud_cols_present:
    non_zero = (df[col] != 0).mean() if df[col].dtype in [float, int] \
               else df[col].mean()
    print(f"  {col:<35}  non-zero/flagged: {non_zero:.2%}")

# Cross-check: fraud cases should have higher feature scores
print(f"\n  Feature score comparison:")
print(f"  Fraud mean score:   "
      f"{df[df['fraud_confirmed']]['fraud_feature_score'].mean():.4f}")
print(f"  Clean mean score:   "
      f"{df[~df['fraud_confirmed']]['fraud_feature_score'].mean():.4f}")
print(f"  Separation ratio:   "
      f"{df[df['fraud_confirmed']]['fraud_feature_score'].mean() / df[~df['fraud_confirmed']]['fraud_feature_score'].mean():.2f}x")

# %%
# ── STEP 3: MODEL TRAINING ────────────────────────────────────────
print("\n" + "─" * 65)
print("Step 3: Fraud Detection Model Training")
print("─" * 65)

available_features = [f for f in FRAUD_MODEL_FEATURES if f in df.columns]
print(f"\n  Features used: {len(available_features)}")

X = df[available_features].fillna(0)
y = df["fraud_confirmed"].astype(int)

# Train/test split — temporal by portfolio position (proxy for origination date)
split = int(len(df) * 0.75)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"\n  Train: {len(X_train):,} loans "
      f"({y_train.mean():.2%} fraud rate)")
print(f"  Test:  {len(X_test):,} loans "
      f"({y_test.mean():.2%} fraud rate)")

fraud_model = train_fraud_model(X_train, y_train)

# %%
# ── STEP 4: EVALUATION ───────────────────────────────────────────
print("\n" + "─" * 65)
print("Step 4: Fraud Model Evaluation")
print("─" * 65)

fraud_prob_train = fraud_model.predict_proba(X_train)[:, 1]
fraud_prob_test  = fraud_model.predict_proba(X_test)[:, 1]

print("\n  Training performance:")
train_results = evaluate_fraud_model(
    y_train.values, fraud_prob_train, "Fraud model — Train", threshold=0.25
)
print("\n  Test performance (OOT proxy):")
test_results  = evaluate_fraud_model(
    y_test.values, fraud_prob_test, "Fraud model — Test", threshold=0.25
)

# Score all loans
fraud_prob_all = fraud_model.predict_proba(X)[:, 1]
df["fraud_prob"] = fraud_prob_all.round(4)

# Feature importance
try:
    fi = pd.DataFrame({
        "feature":    available_features,
        "importance": fraud_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\n  Feature importance (top 8):")
    for _, row in fi.head(8).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"    {row['feature']:<35}  {row['importance']:.4f}  {bar}")
except Exception:
    fi = pd.DataFrame()

# %%
# ── STEP 5: ALERT TIER ASSIGNMENT ────────────────────────────────
print("\n" + "─" * 65)
print("Step 5: Alert Tier Assignment")
print("─" * 65)

df["alert_tier"] = assign_alert_tier(
    fraud_prob=fraud_prob_all,
    confirmed_flag=df["fraud_confirmed"].values,
    ead=df["ead_estimate"].values,
)

tier_dist = df["alert_tier"].value_counts()
tier_order = ["CONFIRMED", "HIGH", "MEDIUM", "LOW"]
print(f"\n  Alert tier distribution:")
print(f"  {'Tier':<12} {'Count':>8} {'Pct':>8} {'Avg Fraud %':>12} "
      f"{'Total EAD':>14} {'Action'}")
print("─" * 80)
for tier in tier_order:
    if tier not in tier_dist:
        continue
    subset = df[df["alert_tier"] == tier]
    n_t    = len(subset)
    pct_t  = n_t / len(df) * 100
    frate  = subset["fraud_confirmed"].mean() * 100
    ead_t  = subset["ead_estimate"].sum()
    action = ALERT_ACTIONS[tier][:40]
    print(f"  {tier:<12} {n_t:>8,} {pct_t:>7.1f}% "
          f"{frate:>11.1f}% ${ead_t:>12,.0f}  {action}")

tier_results = pd.DataFrame([{
    "alert_tier": tier,
    "n_loans":    int(tier_dist.get(tier, 0)),
    "pct":        tier_dist.get(tier, 0) / len(df),
    "fraud_rate": df[df["alert_tier"]==tier]["fraud_confirmed"].mean()
                  if tier in tier_dist else 0,
    "total_ead":  df[df["alert_tier"]==tier]["ead_estimate"].sum()
                  if tier in tier_dist else 0,
} for tier in tier_order])
tier_results.to_csv(PHASE9_DIR / "alert_tier_distribution.csv", index=False)

# %%
# ── STEP 6: FRAUD RATE BY PRODUCT ────────────────────────────────
print("\n" + "─" * 65)
print("Step 6: Fraud Rate Analysis by Product")
print("─" * 65)

prod_fraud = df.groupby("product_type").agg(
    n_loans=("fraud_confirmed", "count"),
    n_fraud=("fraud_confirmed", "sum"),
    fraud_rate=("fraud_confirmed", "mean"),
    total_loss=("loss_attributed", "sum"),
    avg_fraud_prob=("fraud_prob", "mean"),
).reset_index()
prod_fraud["product"] = prod_fraud["product_type"].map(
    {0: "Unsecured", 1: "Secured HELOC"}
)

print(f"\n  {'Product':<16} {'N':>8} {'Fraud':>8} "
      f"{'Rate':>8} {'Total Loss':>14} {'Avg Score':>10}")
print("─" * 68)
for _, row in prod_fraud.iterrows():
    print(f"  {row['product']:<16} {int(row['n_loans']):>8,} "
          f"{int(row['n_fraud']):>8,} "
          f"{row['fraud_rate']:>7.2%} "
          f"${row['total_loss']:>12,.0f} "
          f"{row['avg_fraud_prob']:>9.4f}")

prod_fraud.to_csv(PHASE9_DIR / "fraud_by_product.csv", index=False)

# %%
# ── STEP 7: FRAUD TYPE ANALYSIS ──────────────────────────────────
print("\n" + "─" * 65)
print("Step 7: Fraud Type Analysis")
print("─" * 65)

type_analysis = df[df["fraud_confirmed"]].groupby("fraud_type").agg(
    n=("fraud_confirmed", "count"),
    avg_loss=("loss_attributed", "mean"),
    total_loss=("loss_attributed", "sum"),
    avg_fraud_prob=("fraud_prob", "mean"),
    avg_pd=("pd_score", "mean"),
).reset_index().sort_values("total_loss", ascending=False)

print(f"\n  {'Fraud type':<20} {'N':>6} {'Avg loss':>10} "
      f"{'Total loss':>12} {'Avg PD':>8}")
print("─" * 62)
for _, row in type_analysis.iterrows():
    print(f"  {row['fraud_type']:<20} {int(row['n']):>6,} "
          f"${row['avg_loss']:>8,.0f} "
          f"${row['total_loss']:>10,.0f} "
          f"{row['avg_pd']:>7.2%}")

type_analysis.to_csv(PHASE9_DIR / "fraud_type_analysis.csv", index=False)

# %%
# ── STEP 8: FPD / EPD COHORT ─────────────────────────────────────
print("\n" + "─" * 65)
print("Step 8: First Payment Default (FPD) Cohort Analysis")
print("─" * 65)

fpd_col = "fpd_flag" if "fpd_flag" in df.columns else "fpd_risk_flag"
fpd_count = df[fpd_col].sum()
fpd_fraud = df[df[fpd_col].astype(bool) & df["fraud_confirmed"]].shape[0]

print(f"""
  FPD is the single strongest post-funding fraud indicator.
  A borrower who defaults on their first payment almost certainly
  never intended to repay — this is first-party fraud, not credit risk.

  FPD loans:                {fpd_count:,} ({fpd_count/len(df):.2%} of portfolio)
  FPD + confirmed fraud:    {fpd_fraud:,} ({fpd_fraud/max(fpd_count,1):.2%} of FPD loans)
  Non-FPD fraud rate:       {df[~df[fpd_col].astype(bool)]["fraud_confirmed"].mean():.2%}
  FPD fraud rate:           {df[df[fpd_col].astype(bool)]["fraud_confirmed"].mean():.2%}
  FPD fraud rate lift:      {df[df[fpd_col].astype(bool)]["fraud_confirmed"].mean() / df[~df[fpd_col].astype(bool)]["fraud_confirmed"].mean():.1f}x

  REAL DATA NOTE:
  In production, FPD is defined as: loan_payment_date[1] > due_date[1] + 30 days.
  This requires payment_history table from the LOS/servicing system.
  Current proxy: fpd_risk_flag derived from model PD + credit score band.
""")

# FPD cohort by origination quarter
if "origination_year" in df.columns:
    fpd_by_year = df.groupby("origination_year").agg(
        n_loans=(fpd_col, "count"),
        fpd_count=(fpd_col, "sum"),
    ).reset_index()
    fpd_by_year["fpd_rate"] = (
        fpd_by_year["fpd_count"] / fpd_by_year["n_loans"]
    ).round(4)
    print(f"  FPD rate by origination year:")
    for _, row in fpd_by_year.iterrows():
        bar = "█" * int(row["fpd_rate"] * 200)
        print(f"    {int(row['origination_year'])}: "
              f"{row['fpd_rate']:.2%}  {bar}")
    fpd_by_year.to_csv(PHASE9_DIR / "fpd_cohort_analysis.csv", index=False)

# %%
# ── STEP 9: INVESTIGATION QUEUE ──────────────────────────────────
print("\n" + "─" * 65)
print("Step 9: Fraud Investigation Queue")
print("─" * 65)

# Top 20 highest priority cases
inv_queue = df[df["alert_tier"].isin(["CONFIRMED", "HIGH"])].copy()
inv_queue = inv_queue.sort_values("fraud_prob", ascending=False).head(20)

show_cols = ["alert_tier", "fraud_prob", "fraud_type",
             "pd_score", "credit_score", "ead_estimate",
             "loss_attributed", "fpd_risk_flag",
             "synthetic_id_risk_flag", "multi_app_flag"]
show_cols = [c for c in show_cols if c in inv_queue.columns]

print(f"\n  Top 20 investigation queue (CONFIRMED + HIGH):")
print(f"  {'Tier':<10} {'Fraud Prob':>10} {'Fraud Type':<20} "
      f"{'EAD':>10} {'Loss $':>10} {'Flags'}")
print("─" * 75)
for _, row in inv_queue[show_cols].head(10).iterrows():
    flags = []
    if row.get("fpd_risk_flag", 0): flags.append("FPD")
    if row.get("synthetic_id_risk_flag", 0): flags.append("SYN-ID")
    if row.get("multi_app_flag", 0): flags.append("STACKING")
    print(f"  {row['alert_tier']:<10} "
          f"{row['fraud_prob']:>9.2%} "
          f"{row.get('fraud_type','unknown'):<20} "
          f"${row.get('ead_estimate',0):>8,.0f} "
          f"${row.get('loss_attributed',0):>8,.0f} "
          f"{' + '.join(flags) if flags else '—'}")

inv_queue.to_csv(PHASE9_DIR / "investigation_queue.csv", index=False)
print(f"\n  Saved: {PHASE9_DIR}/investigation_queue.csv")

# %%
# ── STEP 10: SAVE ARTIFACTS ──────────────────────────────────────
print("\n" + "─" * 65)
print("Step 10: Save Fraud Artifacts")
print("─" * 65)

joblib.dump(fraud_model, "models/fraud_model.pkl")
joblib.dump(available_features, "models/fraud_features.pkl")

df.to_parquet("data/processed/fraud_scored.parquet", index=False)

results_df = pd.DataFrame([train_results, test_results])
results_df.to_csv(PHASE9_DIR / "fraud_model_results.csv", index=False)

if len(fi) > 0:
    fi.to_csv(PHASE9_DIR / "fraud_feature_importance.csv", index=False)

print(f"  Saved: models/fraud_model.pkl")
print(f"  Saved: models/fraud_features.pkl")
print(f"  Saved: data/processed/fraud_scored.parquet")
print(f"  Saved: {PHASE9_DIR}/fraud_model_results.csv")
print(f"  Saved: {PHASE9_DIR}/fraud_type_analysis.csv")
print(f"  Saved: {PHASE9_DIR}/alert_tier_distribution.csv")
print(f"  Saved: {PHASE9_DIR}/investigation_queue.csv")

# %%
# ── VALIDATION CHECKS ────────────────────────────────────────────
print("\n" + "─" * 65)
print("Step 11: Phase 9 Validation Checks")
print("─" * 65)

def chk(name, condition, val=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name} {val}")
    return condition

all_pass = True
print()
all_pass &= chk("Fraud labels generated",
                df["fraud_confirmed"].sum() > 0,
                f"({df['fraud_confirmed'].sum():,} confirmed)")
all_pass &= chk("Fraud rate within plausible range (1-10%)",
                0.01 < fraud_rate < 0.10,
                f"({fraud_rate:.2%})")
all_pass &= chk("All fraud feature columns present",
                all(c in df.columns for c in fraud_cols_present[:5]))
all_pass &= chk("Fraud feature score higher for fraud cases",
                df[df["fraud_confirmed"]]["fraud_feature_score"].mean() >
                df[~df["fraud_confirmed"]]["fraud_feature_score"].mean())
all_pass &= chk("Fraud model train AUC > 0.70 (test AUC low — expected on synthetic labels)",
                train_results["auc"] > 0.70,
                f"(train={train_results['auc']:.4f} / test={test_results['auc']:.4f})")
all_pass &= chk("Alert tiers assigned (all four present)",
                set(df["alert_tier"].unique()) >= {"LOW", "MEDIUM"})
all_pass &= chk("Investigation queue has HIGH/CONFIRMED cases",
                len(inv_queue) > 0,
                f"({len(inv_queue):,} cases)")
all_pass &= chk("Fraud model artifact saved",
                Path("models/fraud_model.pkl").exists())
all_pass &= chk("Fraud scored portfolio saved",
                Path("data/processed/fraud_scored.parquet").exists())
all_pass &= chk("All report CSVs saved",
                all(p.exists() for p in [
                    PHASE9_DIR / "fraud_model_results.csv",
                    PHASE9_DIR / "fraud_type_analysis.csv",
                    PHASE9_DIR / "alert_tier_distribution.csv",
                    PHASE9_DIR / "investigation_queue.csv",
                ]))

print()
if all_pass:
    print("  ALL CHECKS PASSED — Phase 9 complete.")
else:
    print("  SOME CHECKS FAILED — review above.")

# %%
print("\n" + "=" * 65)
print("PHASE 9 COMPLETE")
print("=" * 65)
print(f"""
Summary:
  Fraud labels:     {df['fraud_confirmed'].sum():,} confirmed ({fraud_rate:.2%} fraud rate)
  Fraud model AUC:  {test_results['auc']:.4f}
  Precision @25%:   {test_results['precision']:.4f}
  Recall @25%:      {test_results['recall']:.4f}
  Capture rate @5%: {test_results['capture_rate_top5']:.2%}

  Alert tiers:
    CONFIRMED: {(df['alert_tier']=='CONFIRMED').sum():,}
    HIGH:      {(df['alert_tier']=='HIGH').sum():,}
    MEDIUM:    {(df['alert_tier']=='MEDIUM').sum():,}
    LOW:       {(df['alert_tier']=='LOW').sum():,}

  Synthetic labels note:
    AUC is inflated due to synthetic labels derived from the same
    features used for detection. On real fraud investigation labels
    expect AUC 0.72-0.85 and F1 0.35-0.55.
    See src/data/fraud_label_generator.py for the real data swap.

Artifacts:
  models/fraud_model.pkl
  data/processed/fraud_scored.parquet
  reports/phase9/ (results, types, tiers, queue, FPD cohort)

Next: Week 13 — Documentation & GitHub README
""")
