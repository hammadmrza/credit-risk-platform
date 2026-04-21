"""
src/models/pd_model.py
──────────────────────
Probability of Default (PD) modelling module.

Two models are trained:

1. LOGISTIC REGRESSION SCORECARD
   Uses WoE-transformed features. Produces a 300-850 credit score
   via PDO (Points to Double the Odds) calibration. This is the
   traditional credit industry approach — fully interpretable,
   regulator-friendly, and the baseline all other models are
   compared against.

2. XGBOOST PD MODEL
   Primary production model. Gradient boosting consistently achieves
   3-7 AUC points above logistic regression on credit data. Requires
   SHAP for explainability (Phase 5). Evaluated on AUC, KS statistic,
   and Gini coefficient — the three metrics lenders and regulators
   actually use.

EVALUATION METRICS:
  AUC-ROC       Overall discriminatory power (> 0.70 for deployment)
  KS Statistic  Maximum separation between good/bad score distributions
  Gini          = 2 × AUC - 1; the metric lenders formally report
  PSI           Score distribution stability (monitored post-deployment)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import joblib
import logging

log = logging.getLogger(__name__)


# ── Evaluation metrics ────────────────────────────────────────────

class PlattCalibratedModel:
    """
    XGBoost model wrapped with Platt scaling calibration.
    Defined at module level for joblib pickling compatibility.
    """
    def __init__(self, base, scaler):
        self.base    = base
        self.scaler  = scaler
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self.scaler.predict_proba(raw)
        return cal

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return getattr(self.base, "feature_importances_", None)

    @property
    def best_iteration(self):
        return getattr(self.base, "best_iteration", None)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    KS (Kolmogorov-Smirnov) statistic.
    Maximum separation between cumulative good and bad distributions.
    Industry threshold for deployment: KS > 0.30
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Gini = 2 × AUC - 1.
    The metric lenders formally report in model validation docs.
    Range: 0 (random) to 1 (perfect).
    """
    auc = roc_auc_score(y_true, y_prob)
    return float(2 * auc - 1)


def evaluate_model(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   model_name: str = "Model") -> dict:
    """Full evaluation suite: AUC, KS, Gini."""
    auc  = roc_auc_score(y_true, y_prob)
    ks   = ks_statistic(y_true, y_prob)
    gini = gini_coefficient(y_true, y_prob)

    results = {
        "model":    model_name,
        "auc":      round(auc, 4),
        "ks":       round(ks, 4),
        "gini":     round(gini, 4),
        "n_obs":    len(y_true),
        "bad_rate": round(float(y_true.mean()), 4),
    }

    log.info(f"\n{'─'*50}")
    log.info(f"  {model_name}")
    log.info(f"  AUC:   {auc:.4f}  {'✓' if auc > 0.70 else '⚠ below 0.70'}")
    log.info(f"  KS:    {ks:.4f}   {'✓' if ks > 0.30 else '⚠ below 0.30'}")
    log.info(f"  Gini:  {gini:.4f}  {'✓' if gini > 0.40 else '⚠ below 0.40'}")
    log.info(f"{'─'*50}")
    return results


def evaluate_by_segment(y_true: pd.Series,
                         y_prob: np.ndarray,
                         segment: pd.Series,
                         model_name: str = "Model") -> pd.DataFrame:
    """
    Evaluate model performance separately for each product segment.
    Segmented AUC reveals whether the model performs equally well
    across secured and unsecured products.
    """
    rows = []
    for seg_val in sorted(segment.unique()):
        mask = segment == seg_val
        if mask.sum() < 50:
            continue
        seg_name = "Unsecured" if seg_val == 0 else "Secured"
        try:
            auc  = roc_auc_score(y_true[mask], y_prob[mask])
            ks   = ks_statistic(y_true[mask].values, y_prob[mask])
            gini = gini_coefficient(y_true[mask].values, y_prob[mask])
            rows.append({
                "segment": seg_name,
                "n":       int(mask.sum()),
                "bad_rate": round(float(y_true[mask].mean()), 4),
                "auc":     round(auc, 4),
                "ks":      round(ks, 4),
                "gini":    round(gini, 4),
                "model":   model_name,
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


# ── Score calibration ─────────────────────────────────────────────

def pd_to_score(pd_values: np.ndarray,
                pdo: int = 20,
                base_score: int = 600,
                base_odds: float = 4.0) -> np.ndarray:
    # base_odds=4.0 calibrated to LendingClub actual default rate ~20%
    # (4 good borrowers per 1 bad at the base score of 600)
    # base_odds=50 was for prime-only populations and compresses all LC scores to Tier E
    """
    Convert PD to credit score using PDO calibration.
    PDO = Points to Double the Odds.
    """
    pd_values = np.clip(pd_values, 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    odds   = (1 - pd_values) / pd_values
    scores = offset + factor * np.log(odds)
    return np.clip(np.round(scores).astype(int), 300, 850)


def assign_risk_tier(scores: np.ndarray) -> np.ndarray:
    """Assign A-E risk tier from credit scores."""
    return np.where(scores >= 720, "A",
           np.where(scores >= 680, "B",
           np.where(scores >= 630, "C",
           np.where(scores >= 580, "D", "E"))))


# ── Model 1: Logistic Regression Scorecard ────────────────────────

def train_scorecard(X_train_woe: pd.DataFrame,
                    y_train: pd.Series,
                    C: float = 0.1) -> Pipeline:
    """
    Train logistic regression on WoE-transformed features.

    Args:
        X_train_woe: WoE-transformed feature matrix.
        y_train: Binary target.
        C: Regularisation strength (lower = more regularised).

    Returns:
        Fitted sklearn Pipeline (StandardScaler + LogisticRegression).
    """
    log.info("Training Logistic Regression Scorecard ...")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C,
            max_iter=500,
            random_state=42,
            solver="lbfgs",
            class_weight="balanced"
        ))
    ])
    pipe.fit(X_train_woe, y_train)
    log.info("  Scorecard training complete.")
    return pipe


# ── Model 2: XGBoost PD Model ─────────────────────────────────────

def train_xgboost(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   use_optuna: bool = False,
                   n_trials: int = 30,
                   calibrate: bool = True) -> object:
    """
    Train XGBoost gradient boosting PD model with Platt scaling calibration.

    FIX: Added proper calibration layer and validation fold.
    Raw XGBoost scores are ranking scores, not calibrated probabilities.
    Without calibration, mean PD ≈ 0.46 on synthetic data vs true 0.24
    default rate — a 22pp miscalibration that breaks IFRS 9 staging,
    ECL coverage ratios, and risk-based pricing.

    Platt scaling fits a logistic regression on a held-out calibration
    fold to map raw XGBoost scores to true default probabilities.

    Args:
        X_train: Raw feature matrix (not WoE).
        y_train: Binary target.
        use_optuna: Run hyperparameter search with Optuna.
        n_trials: Number of Optuna trials if use_optuna=True.
        calibrate: Apply Platt scaling calibration (recommended: True).

    Returns:
        Fitted model (CalibratedClassifierCV if calibrate=True,
        else raw XGBClassifier).
    """
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split as tts

    # ── Split: 85% train XGBoost, 15% calibration fold ───────────
    if calibrate:
        X_tr, X_cal, y_tr, y_cal = tts(
            X_train, y_train,
            test_size=0.15,
            stratify=y_train,
            random_state=42
        )
        log.info(f"  Train/calibration split: "
                 f"{len(X_tr):,} / {len(X_cal):,}")
    else:
        X_tr, y_tr = X_train, y_train

    scale_pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    log.info(f"  scale_pos_weight = {scale_pos_weight:.2f}")

    if use_optuna:
        log.info("Running Optuna hyperparameter search ...")
        params = _optuna_xgb_search(X_tr, y_tr, scale_pos_weight, n_trials)
    else:
        params = {
            "n_estimators":     300,
            "max_depth":        4,
            "learning_rate":    0.05,
            "subsample":        0.80,
            "colsample_bytree": 0.80,
            "min_child_weight": 20,
            "reg_alpha":        0.1,
            "reg_lambda":       1.0,
            "scale_pos_weight": scale_pos_weight,
            "random_state":     42,
            "eval_metric":      "auc",
            "use_label_encoder": False,
        }

    # ── Train XGBoost ─────────────────────────────────────────────
    base_model = xgb.XGBClassifier(**params)

    if calibrate:
        # Use validation set for early stopping
        base_model.set_params(n_estimators=500, early_stopping_rounds=20)
        base_model.fit(
            X_tr, y_tr,
            eval_set=[(X_cal, y_cal)],
            verbose=False
        )
        log.info(f"  Best iteration: {base_model.best_iteration}")

        # ── Platt scaling calibration ─────────────────────────────
        log.info("  Applying Platt scaling calibration ...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Get raw scores on calibration fold
        raw_cal_scores = base_model.predict_proba(X_cal)[:, 1].reshape(-1, 1)

        # Fit logistic regression (Platt scaling) on raw scores → true labels
        platt_scaler   = LogisticRegression(C=1.0, random_state=42)
        platt_scaler.fit(raw_cal_scores, y_cal)

        calibrated = PlattCalibratedModel(base_model, platt_scaler)

        # Verify calibration improved
        raw_mean  = base_model.predict_proba(X_cal)[:, 1].mean()
        cal_mean  = calibrated.predict_proba(X_cal)[:, 1].mean()
        true_rate = float(y_cal.mean())
        log.info(f"  Calibration check (cal fold):")
        log.info(f"    True default rate: {true_rate:.4f}")
        log.info(f"    Raw XGB mean PD:   {raw_mean:.4f}  "
                 f"(gap: {raw_mean - true_rate:+.4f})")
        log.info(f"    Calibrated PD:     {cal_mean:.4f}  "
                 f"(gap: {cal_mean - true_rate:+.4f})")

        log.info("  XGBoost + Platt calibration complete.")
        return calibrated

    else:
        base_model.fit(X_tr, y_tr,
                       eval_set=[(X_tr, y_tr)],
                       verbose=False)
        log.info("  XGBoost training complete (uncalibrated).")
        return base_model


def _optuna_xgb_search(X_train, y_train,
                        scale_pos_weight, n_trials=30):
    """Optuna hyperparameter search for XGBoost."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "max_depth":        trial.suggest_int("max_depth", 3, 6),
                "learning_rate":    trial.suggest_float("lr", 0.01, 0.15,
                                                         log=True),
                "subsample":        trial.suggest_float("sub", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("col", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("mcw", 5, 50),
                "reg_alpha":        trial.suggest_float("alpha", 1e-3, 1.0,
                                                         log=True),
                "scale_pos_weight": scale_pos_weight,
                "random_state":     42,
                "use_label_encoder": False,
                "eval_metric":      "auc",
            }
            clf = xgb.XGBClassifier(**params)
            scores = cross_val_score(
                clf, X_train, y_train,
                cv=StratifiedKFold(3, shuffle=True, random_state=42),
                scoring="roc_auc", n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        log.info(f"  Best AUC (CV): {study.best_value:.4f}")
        log.info(f"  Best params: {best}")

        return {
            "n_estimators":     best["n_estimators"],
            "max_depth":        best["max_depth"],
            "learning_rate":    best["lr"],
            "subsample":        best["sub"],
            "colsample_bytree": best["col"],
            "min_child_weight": best["mcw"],
            "reg_alpha":        best["alpha"],
            "scale_pos_weight": scale_pos_weight,
            "random_state":     42,
            "use_label_encoder": False,
            "eval_metric":      "auc",
        }
    except ImportError:
        log.warning("Optuna not available — using default params")
        return {}


# ── Separate product model training (v1.1) ─────────────────────
def train_segmented_models(X_train_unsecured, y_train_unsecured,
                            X_train_secured,   y_train_secured,
                            X_cal_unsecured,   y_cal_unsecured,
                            X_cal_secured,     y_cal_secured,
                            params=None):
    """
    Train separate PD models for unsecured and secured (HELOC) products.
    This removes the loan_term_months product proxy from both models
    and is expected to push combined AUC to 0.75-0.82.

    Returns:
        model_unsecured: Calibrated XGBoost for unsecured loans
        model_secured:   Calibrated XGBoost for HELOC loans
    """
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression

    default_params = {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "use_label_encoder": False,
        "eval_metric": "auc", "random_state": 42,
    }
    if params:
        default_params.update(params)

    models = {}
    for name, X_tr, y_tr, X_cal, y_cal in [
        ("unsecured", X_train_unsecured, y_train_unsecured, X_cal_unsecured, y_cal_unsecured),
        ("secured",   X_train_secured,   y_train_secured,   X_cal_secured,   y_cal_secured),
    ]:
        spw = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
        xgb = XGBClassifier(**{**default_params, "scale_pos_weight": spw})
        xgb.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)],
                verbose=False, early_stopping_rounds=30)

        # Platt calibration
        raw_probs = xgb.predict_proba(X_cal)[:, 1]
        platt = LogisticRegression(C=1e5)
        platt.fit(raw_probs.reshape(-1, 1), y_cal)

        class CalibratedModel:
            def __init__(self, base, cal):
                self.base = base; self.cal = cal
                self.classes_ = np.array([0, 1])
            def predict_proba(self, X):
                raw = self.base.predict_proba(X)[:, 1]
                cal = self.cal.predict_proba(raw.reshape(-1,1))[:, 1]
                return np.column_stack([1-cal, cal])

        models[name] = CalibratedModel(xgb, platt)
        print(f"  {name} model trained and calibrated.")

    return models["unsecured"], models["secured"]
