"""
src/models/pd_model_segmented.py
─────────────────────────────────
Per-product PD models (v1.1) — trains separate XGBoost models for
Unsecured (LendingClub) and Secured (HELOC) books.

RATIONALE
─────────
The v1.0 unified model trains one XGBoost on the harmonized dual-product
portfolio with `product_type` as a feature. This has two limitations:

  1. `loan_term_months` acts as a product-type proxy (36/60 for unsecured
     vs. imputed-36 for HELOC), inflating its apparent importance and
     depressing the reported combined AUC to ~0.68.

  2. The two books have genuinely different default mechanisms:
       - Unsecured: driven by DTI, credit_score, inquiries, utilization
       - Secured (HELOC): driven by external_risk_estimate,
         pct_trades_never_delinquent, num_high_utilization_trades
     A single model forces shared feature splits across both populations,
     limiting achievable discrimination on each segment.

WHY WE STILL KEEP THE UNIFIED MODEL
────────────────────────────────────
Operational cost: two models require two WoE binners, two calibrators,
two SHAP explainers, two monitoring pipelines, two OSFI E-23 model
cards. Accept that cost only if the AUC lift justifies it.

HELOC thin-data problem: ~10K HELOC rows vs 2.2M LendingClub rows.
HELOC alone risks overfitting. The unified model lets HELOC borrow
signal from LendingClub for features that generalize (credit_score
behaviour, DTI bands).

Cross-product comparability: with one model, scores are directly
comparable across products. With two, "720 unsecured" and "720 HELOC"
are calibrated to different baseline default rates and cannot be
compared directly. This matters for portfolio concentration limits.

DECISION (v1.1): offer BOTH — unified for cross-product decisioning
and reporting, segmented for per-product monitoring and validation.

USAGE
─────
    from src.models.pd_model_segmented import train_segmented_models
    results = train_segmented_models(
        X_train, y_train, X_test, y_test, product_col="product_type"
    )
    # results = {"unsecured": {model, metrics}, "secured": {model, metrics}}
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# Features to exclude from segmented models — these are product proxies
# or interaction features that only make sense in the combined model.
SEGMENTED_EXCLUDE = {
    "product_type",        # Single product per model — not a feature
    "loan_term_months",    # Product proxy; for unsecured still informative
                           #   but kept out by default to make segmented
                           #   models strictly credit-feature driven.
                           #   Remove from this set if term is product-native.
    "ltv_x_product",       # Product interaction
    "score_x_product",     # Product interaction
}


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """AUC, KS, Gini from probability predictions."""
    if len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "ks": np.nan, "gini": np.nan,
                "n": len(y_true), "default_rate": float(y_true.mean())}

    auc = roc_auc_score(y_true, y_pred)
    # KS = max |TPR - FPR| across thresholds
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_neg = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    ks = float(np.abs(cum_pos - cum_neg).max())
    gini = 2 * auc - 1
    return {
        "auc":  round(float(auc),  4),
        "ks":   round(ks,          4),
        "gini": round(float(gini), 4),
        "n":    int(len(y_true)),
        "default_rate": round(float(y_true.mean()), 4),
    }


def _train_one_product(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    product_name: str,
) -> dict:
    """
    Train a single XGBoost PD model on a filtered (single-product) slice.

    Returns dict with model, test_metrics, train_metrics, features_used.
    """
    # Drop product-interaction features that don't make sense within one segment
    features = [c for c in X_train.columns if c not in SEGMENTED_EXCLUDE]
    Xtr = X_train[features].fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)
    Xte = X_test [features].fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)

    if not HAS_XGB:
        raise RuntimeError(
            "xgboost not installed. Install with: pip install xgboost"
        )

    # HELOC is small — reduce max_depth and add regularisation to prevent overfit
    is_small = len(Xtr) < 50_000
    params = dict(
        n_estimators=200 if is_small else 400,
        max_depth=4 if is_small else 6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20 if is_small else 10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

    log.info(f"  Training {product_name} XGBoost on {len(Xtr):,} rows × "
             f"{len(features)} features (default rate "
             f"{float(y_train.mean()):.2%})")
    model = xgb.XGBClassifier(**params)
    model.fit(Xtr, y_train.values,
              eval_set=[(Xte, y_test.values)],
              verbose=False)

    pred_train = model.predict_proba(Xtr)[:, 1]
    pred_test  = model.predict_proba(Xte)[:, 1]

    return {
        "model": model,
        "features": features,
        "train_metrics": _compute_metrics(y_train.values, pred_train),
        "test_metrics":  _compute_metrics(y_test .values, pred_test),
    }


def train_segmented_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    product_col: str = "product_type",
) -> dict:
    """
    Train separate PD models for each product segment.

    Args:
        X_train, y_train: combined training data (with product_col column)
        X_test, y_test:   combined OOT test data (with product_col column)
        product_col:      column identifying product (0=Unsecured, 1=Secured)

    Returns:
        dict with keys "unsecured", "secured" — each containing
        {model, features, train_metrics, test_metrics}.
    """
    if product_col not in X_train.columns:
        raise ValueError(f"Product column '{product_col}' not in X_train")

    results = {}

    for product_int, product_name in [(0, "unsecured"), (1, "secured")]:
        tr_mask = X_train[product_col] == product_int
        te_mask = X_test [product_col] == product_int

        n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
        if n_tr < 500 or n_te < 100:
            log.warning(f"  Skipping {product_name}: insufficient data "
                        f"(train={n_tr}, test={n_te})")
            results[product_name] = {
                "model": None, "features": [], "train_metrics": {}, "test_metrics": {},
                "skipped": True, "reason": f"insufficient data (train={n_tr}, test={n_te})",
            }
            continue

        log.info(f"▶  Segment: {product_name.upper()}")
        results[product_name] = _train_one_product(
            X_train[tr_mask], y_train[tr_mask],
            X_test [te_mask], y_test [te_mask],
            product_name,
        )

    return results


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """
    Build a comparison DataFrame from segmented training results.
    One row per (product, dataset) with AUC/KS/Gini + sample size.
    """
    rows = []
    for product_name, res in results.items():
        if res.get("skipped"):
            continue
        for split_name, key in [("Train", "train_metrics"), ("Test (OOT)", "test_metrics")]:
            m = res.get(key) or {}
            if not m:
                continue
            rows.append({
                "product":       product_name.title(),
                "split":         split_name,
                "n":             m.get("n", 0),
                "default_rate":  m.get("default_rate", np.nan),
                "auc":           m.get("auc", np.nan),
                "ks":            m.get("ks",  np.nan),
                "gini":          m.get("gini", np.nan),
            })
    return pd.DataFrame(rows)


def save_artifacts(results: dict,
                   models_dir: Path = Path("models"),
                   reports_dir: Path = Path("reports/phase4")) -> None:
    """Persist segmented models and comparison CSV to disk."""
    import joblib
    models_dir .mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for product_name, res in results.items():
        if res.get("skipped") or res.get("model") is None:
            continue
        out = models_dir / f"xgb_pd_{product_name}.pkl"
        joblib.dump(res["model"], out)
        log.info(f"  Saved {out}")
        feat_out = models_dir / f"xgb_pd_{product_name}_features.pkl"
        joblib.dump(res["features"], feat_out)

    df = results_to_dataframe(results)
    out_csv = reports_dir / "model_comparison_segmented.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  Saved {out_csv}")
