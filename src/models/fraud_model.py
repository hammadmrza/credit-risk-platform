"""
src/models/fraud_model.py
──────────────────────────
Fraud detection model: XGBoost classifier + alert tier assignment.

TWO-STAGE DETECTION:
  Stage 1: XGBoost fraud probability model
    Input:  application features + fraud-specific features
    Output: fraud probability score (0-1)

  Stage 2: Alert tier assignment
    Input:  fraud probability + contextual flags
    Output: LOW / MEDIUM / HIGH / CONFIRMED alert tier
            + recommended investigation action

ALERT TIER DEFINITIONS:
  LOW        PD < 0.10  — Routine monitoring. No action required.
  MEDIUM     PD 0.10-0.25 — Enhanced monitoring. Flag for QA review.
  HIGH       PD 0.25-0.50 — Escalate to fraud team. Hold funding.
  CONFIRMED  PD > 0.50 OR confirmed_flag=True — Immediate investigation.

EVALUATION METRICS FOR FRAUD MODELS:
  Unlike credit models (AUC / KS / Gini), fraud models use:
  Precision:  Of all alerts, what % are real fraud?
              (Low precision = analyst fatigue from false positives)
  Recall:     Of all fraud, what % did we catch?
              (Low recall = fraud slipping through)
  F1 Score:   Harmonic mean of precision and recall
  KS:         Separation between fraud and non-fraud score distributions
  Fraud Capture Rate @ Top 5%:
              What % of confirmed fraud falls in the top 5% of scores?
              Industry target: > 50% fraud captured in top 5% of alerts.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import logging

log = logging.getLogger(__name__)

ALERT_THRESHOLDS = {
    "LOW":       0.10,
    "MEDIUM":    0.25,
    "HIGH":      0.50,
    "CONFIRMED": 1.01,  # Sentinel — never exceeded
}

FRAUD_MODEL_FEATURES = [
    "pd_score",
    "credit_score",
    "fpd_risk_score",
    "velocity_score",
    "synthetic_id_score",
    "income_loan_ratio",
    "high_loan_to_income",
    "short_tenure_flag",
    "score_income_inconsistency",
    "high_velocity_flag",
    "multi_app_flag",
    "synthetic_id_risk_flag",
    "address_mismatch_flag",
    "fraud_feature_score",
    "product_type",
    "lgd_estimate",
    "ead_estimate",
]


def train_fraud_model(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       use_optuna: bool = False) -> object:
    """
    Train XGBoost fraud detection model.

    NOTE ON LABEL QUALITY:
    Training on synthetic labels will produce optimistically high
    AUC/precision because the labels are derived from the same
    features used for detection. On real fraud labels from
    investigations, expect AUC 0.72-0.85 and F1 0.35-0.55
    (fraud detection is genuinely harder than credit scoring).

    Args:
        X_train: Feature matrix.
        y_train: Binary fraud label (1=fraud, 0=clean).
        use_optuna: Run hyperparameter search.

    Returns:
        Fitted XGBoost classifier.
    """
    import xgboost as xgb

    fraud_rate = y_train.mean()
    scale_pos  = float((y_train == 0).sum() / (y_train == 1).sum())

    log.info(f"Training fraud model ...")
    log.info(f"  Training samples: {len(y_train):,}")
    log.info(f"  Fraud rate:       {fraud_rate:.2%}")
    log.info(f"  Scale pos weight: {scale_pos:.1f}")

    params = {
        "n_estimators":     200,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 10,
        "scale_pos_weight": scale_pos,
        "random_state":     42,
        "eval_metric":      "aucpr",
        "use_label_encoder": False,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    log.info("  Fraud model training complete.")
    return model


def evaluate_fraud_model(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          model_name: str = "Fraud Model",
                          threshold: float = 0.25) -> dict:
    """
    Evaluate fraud model with fraud-specific metrics.

    Args:
        y_true: True fraud labels (0/1).
        y_prob: Fraud probability scores.
        model_name: Label for logging.
        threshold: Decision threshold for binary metrics.

    Returns:
        Dict with AUC, precision, recall, F1, KS,
        fraud capture rate at top 5%.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Standard metrics
    auc = roc_auc_score(y_true, y_prob)

    # Fraud-specific: precision/recall at threshold
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    # KS statistic
    from scipy.stats import ks_2samp
    fraud_scores = y_prob[y_true == 1]
    clean_scores = y_prob[y_true == 0]
    ks_stat      = float(ks_2samp(fraud_scores, clean_scores).statistic)

    # Fraud capture rate @ top 5% of scores
    top5_cutoff    = np.percentile(y_prob, 95)
    in_top5        = y_prob >= top5_cutoff
    fraud_in_top5  = (y_true[in_top5] == 1).sum()
    total_fraud    = (y_true == 1).sum()
    capture_rate_5 = fraud_in_top5 / max(total_fraud, 1)

    # Fraud capture rate @ top 10%
    top10_cutoff   = np.percentile(y_prob, 90)
    in_top10       = y_prob >= top10_cutoff
    fraud_in_top10 = (y_true[in_top10] == 1).sum()
    capture_rate_10 = fraud_in_top10 / max(total_fraud, 1)

    results = {
        "model":             model_name,
        "auc":               round(auc, 4),
        "precision":         round(prec, 4),
        "recall":            round(rec, 4),
        "f1":                round(f1, 4),
        "ks":                round(ks_stat, 4),
        "capture_rate_top5": round(capture_rate_5, 4),
        "capture_rate_top10":round(capture_rate_10, 4),
        "threshold":         threshold,
        "n_alerts":          int(y_pred.sum()),
        "alert_rate":        round(y_pred.mean(), 4),
        "n_fraud_total":     int(total_fraud),
    }

    log.info(f"\n{'─'*55}")
    log.info(f"  {model_name}")
    log.info(f"  AUC:                {auc:.4f}")
    log.info(f"  Precision @ {threshold:.0%}:   {prec:.4f}  "
             f"(of alerts, this % are real fraud)")
    log.info(f"  Recall @ {threshold:.0%}:      {rec:.4f}  "
             f"(of fraud, this % was caught)")
    log.info(f"  F1 Score:           {f1:.4f}")
    log.info(f"  KS:                 {ks_stat:.4f}")
    log.info(f"  Capture rate top 5%: {capture_rate_5:.2%}")
    log.info(f"  Alert rate:         {y_pred.mean():.2%} "
             f"({y_pred.sum():,} alerts)")
    log.info(f"{'─'*55}")

    return results


def assign_alert_tier(fraud_prob: np.ndarray,
                       confirmed_flag: np.ndarray = None,
                       high_value_flag: np.ndarray = None,
                       ead: np.ndarray = None) -> np.ndarray:
    """
    Assign alert tier to each loan based on fraud probability
    and contextual risk factors.

    Args:
        fraud_prob: Fraud probability scores (0-1).
        confirmed_flag: Pre-confirmed fraud indicator.
        high_value_flag: High EAD loans (elevated priority).
        ead: Exposure at Default (for value-weighted tiers).

    Returns:
        Array of alert tier strings.
    """
    tiers = np.where(fraud_prob < 0.10, "LOW",
            np.where(fraud_prob < 0.25, "MEDIUM",
            np.where(fraud_prob < 0.50, "HIGH", "CONFIRMED")))

    # Escalate high-value loans one tier
    if ead is not None and high_value_flag is None:
        high_value_flag = ead > 25000

    if high_value_flag is not None:
        tiers = np.where(
            high_value_flag & (tiers == "MEDIUM"), "HIGH",
            np.where(high_value_flag & (tiers == "HIGH"), "CONFIRMED",
                     tiers)
        )

    # Confirmed fraud overrides tier
    if confirmed_flag is not None:
        tiers = np.where(confirmed_flag, "CONFIRMED", tiers)

    return tiers


ALERT_ACTIONS = {
    "LOW":       "Routine monitoring. Include in monthly fraud review.",
    "MEDIUM":    "Enhanced monitoring. QA review before next payment cycle.",
    "HIGH":      "Escalate to fraud team. Consider funding hold.",
    "CONFIRMED": "Immediate investigation. Freeze account. Document for recovery.",
}
