"""
src/features/woe_binning.py
────────────────────────────
Weight of Evidence (WoE) binning and Information Value (IV) scoring.

WoE binning is the foundation of traditional credit scorecard development.
It converts continuous variables into categorical risk bands, with each
band assigned a WoE score based on the ratio of good:bad borrowers in it.

WHY WoE BINNING MATTERS IN CREDIT:
  1. Handles non-linear relationships between features and default
  2. Naturally handles outliers (extreme values fall into boundary bins)
  3. Produces interpretable, regulator-friendly risk bands
  4. Enables IV-based feature selection (drop low-signal features)
  5. Required for the logistic regression scorecard (Phase 4)

INFORMATION VALUE INTERPRETATION:
  IV < 0.02  → Useless predictor — drop
  IV 0.02–0.1 → Weak predictor
  IV 0.1–0.3 → Medium predictor
  IV 0.3–0.5 → Strong predictor
  IV > 0.5   → Suspicious (possible data leakage — investigate)

PDO SCORE CALIBRATION:
  After WoE → logistic regression, we convert log-odds to a credit score
  using the Points-to-Double-Odds (PDO) formula. This produces a score
  on the familiar 300-850 scale where:
    Higher score = lower risk = better borrower
    20 PDO = every 20-point increase doubles the odds of being good
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

# Features to bin — selected based on Phase 2 correlation analysis
# Ordered by expected IV (highest first)
BINNING_FEATURES = [
    # Primary predictors
    "credit_score",
    "external_risk_estimate",
    "pct_trades_never_delinquent",
    "annual_income",
    "dti",
    "credit_utilization",
    # Delinquency signals
    "months_since_recent_delinquency",
    "num_derogatory_marks",
    "num_inquiries_last_6m",
    # Account history
    "months_since_oldest_trade",
    "total_accounts",
    "num_high_utilization_trades",
    # Secured-specific
    "ltv_ratio",
    # Alternative data
    "alt_data_score",
    # Employment / loan
    "employment_length_years",
    "loan_amount",
    "loan_term_months",
]

IV_THRESHOLD_DROP   = 0.02   # Drop features below this
IV_THRESHOLD_WEAK   = 0.10
IV_THRESHOLD_MEDIUM = 0.30
IV_THRESHOLD_STRONG = 0.50


def compute_iv_manual(df: pd.DataFrame,
                       feature: str,
                       target: str = "default_flag",
                       bins: int = 10) -> float:
    """
    Compute Information Value for a single feature using equal-frequency bins.
    Fallback when optbinning is not available.
    """
    try:
        work = df[[feature, target]].dropna()
        if len(work) < 100:
            return 0.0

        work["bin"] = pd.qcut(work[feature], q=bins,
                               duplicates="drop")
        grouped = work.groupby("bin")[target].agg(
            n_bad="sum", n_total="count"
        )
        grouped["n_good"] = grouped["n_total"] - grouped["n_bad"]

        total_bad  = grouped["n_bad"].sum()
        total_good = grouped["n_good"].sum()

        if total_bad == 0 or total_good == 0:
            return 0.0

        grouped["pct_bad"]  = grouped["n_bad"]  / total_bad
        grouped["pct_good"] = grouped["n_good"] / total_good

        # Avoid log(0)
        grouped = grouped[
            (grouped["pct_bad"] > 0) & (grouped["pct_good"] > 0)
        ]

        grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])
        grouped["iv"]  = (grouped["pct_good"] - grouped["pct_bad"]) * grouped["woe"]

        return grouped["iv"].sum()

    except Exception:
        return 0.0


def run_woe_binning(df_train: pd.DataFrame,
                    df_test: pd.DataFrame,
                    target: str = "default_flag",
                    features: Optional[List[str]] = None,
                    use_optbinning: bool = True
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """
    Run WoE binning on training data and apply to test data.

    Args:
        df_train: Training DataFrame (after imputation).
        df_test: Test DataFrame (after imputation).
        target: Binary target column.
        features: Features to bin. Defaults to BINNING_FEATURES.
        use_optbinning: Use optbinning library if available.

    Returns:
        (df_train_woe, df_test_woe, iv_table, binning_process)
    """
    if features is None:
        features = [f for f in BINNING_FEATURES if f in df_train.columns]

    log.info(f"Running WoE binning on {len(features)} features ...")

    iv_records = []
    woe_train_cols = {}
    woe_test_cols  = {}
    binning_process = None

    # ── Try optbinning first (preferred) ─────────────────────────
    optbinning_available = False
    try:
        from optbinning import BinningProcess
        optbinning_available = True and use_optbinning
    except ImportError:
        log.warning("optbinning not installed — using manual WoE calculation")
        log.warning("Install with: pip install optbinning --break-system-packages")

    if optbinning_available:
        try:
            log.info("Using optbinning BinningProcess ...")

            # Special codes for sentinel values
            special_codes = [
                SENTINEL_NOT_APPLICABLE := -1.0,
                999.0,   # never-delinquent sentinel
            ]

            # Per-feature binning parameters
            binning_fit_params = {}
            for feat in features:
                params = {"max_n_bins": 8, "min_bin_size": 0.05}
                # Monotonicity constraints for key features
                if feat in ["credit_score", "annual_income",
                            "pct_trades_never_delinquent",
                            "external_risk_estimate",
                            "months_since_recent_delinquency"]:
                    params["monotonic_trend"] = "descending"
                elif feat in ["dti", "credit_utilization", "ltv_ratio",
                              "num_derogatory_marks", "num_inquiries_last_6m",
                              "num_high_utilization_trades"]:
                    params["monotonic_trend"] = "ascending"
                binning_fit_params[feat] = params

            bp = BinningProcess(
                variable_names=features,
                special_codes=special_codes,
                binning_fit_params=binning_fit_params
            )

            X_train = df_train[features]
            y_train = df_train[target]

            bp.fit(X_train, y_train)
            binning_process = bp

            # Transform
            X_train_woe = bp.transform(X_train, metric="woe")
            X_test_woe  = bp.transform(df_test[features], metric="woe")

            # Rename columns
            woe_cols = {f: f"{f}_woe" for f in features}
            X_train_woe = X_train_woe.rename(columns=woe_cols)
            X_test_woe  = X_test_woe.rename(columns=woe_cols)

            # Get IV table from binning process
            try:
                summary = bp.summary()
                for feat in features:
                    try:
                        binning_table = bp.get_binned_variable(feat).binning_table
                        iv_val = binning_table.build()["IV"].iloc[-1]
                    except Exception:
                        iv_val = compute_iv_manual(df_train, feat)
                    iv_records.append({
                        "feature": feat,
                        "iv": round(float(iv_val), 4),
                        "method": "optbinning"
                    })
            except Exception:
                # Fall back to manual IV if summary fails
                for feat in features:
                    iv_val = compute_iv_manual(df_train, feat)
                    iv_records.append({
                        "feature": feat,
                        "iv": round(iv_val, 4),
                        "method": "manual_fallback"
                    })

            # Add WoE columns to dataframes
            df_train_woe = pd.concat(
                [df_train.reset_index(drop=True),
                 X_train_woe.reset_index(drop=True)], axis=1
            )
            df_test_woe = pd.concat(
                [df_test.reset_index(drop=True),
                 X_test_woe.reset_index(drop=True)], axis=1
            )

            log.info("optbinning WoE transformation complete.")

        except Exception as e:
            log.warning(f"optbinning failed ({e}) — falling back to manual")
            optbinning_available = False

    # ── Manual WoE fallback ───────────────────────────────────────
    if not optbinning_available:
        log.info("Computing WoE manually ...")
        df_train_woe = df_train.copy()
        df_test_woe  = df_test.copy()

        for feat in features:
            if feat not in df_train.columns:
                continue
            try:
                iv_val = compute_iv_manual(df_train, feat)
                iv_records.append({
                    "feature": feat,
                    "iv": round(iv_val, 4),
                    "method": "manual"
                })

                # Create simple decile bins as WoE proxy
                work = df_train[[feat, target]].dropna()
                try:
                    _, bin_edges = pd.qcut(
                        work[feat], q=10,
                        duplicates="drop", retbins=True
                    )

                    def apply_bin_woe(series, edges):
                        binned = pd.cut(series, bins=edges,
                                        include_lowest=True)
                        return binned.cat.codes.astype(float)

                    df_train_woe[f"{feat}_woe"] = apply_bin_woe(
                        df_train[feat], bin_edges
                    )
                    df_test_woe[f"{feat}_woe"] = apply_bin_woe(
                        df_test[feat], bin_edges
                    )
                except Exception:
                    df_train_woe[f"{feat}_woe"] = df_train[feat]
                    df_test_woe[f"{feat}_woe"]  = df_test[feat]

            except Exception as e:
                log.warning(f"  Skipping {feat}: {e}")
                iv_records.append({"feature": feat, "iv": 0.0,
                                   "method": "failed"})

    # ── Build IV table ────────────────────────────────────────────
    iv_table = pd.DataFrame(iv_records).sort_values(
        "iv", ascending=False
    ).reset_index(drop=True)

    # Add IV interpretation
    def iv_label(iv):
        if iv < IV_THRESHOLD_DROP:   return "USELESS — drop"
        elif iv < IV_THRESHOLD_WEAK: return "Weak"
        elif iv < IV_THRESHOLD_MEDIUM: return "Medium"
        elif iv < IV_THRESHOLD_STRONG: return "Strong"
        else: return "Very strong (check for leakage)"

    iv_table["strength"] = iv_table["iv"].map(iv_label)

    # Features to keep (IV >= threshold)
    features_to_keep = iv_table[
        iv_table["iv"] >= IV_THRESHOLD_DROP
    ]["feature"].tolist()

    log.info(f"\nIV Summary:")
    log.info(f"  Total features evaluated: {len(iv_table)}")
    log.info(f"  Features to keep (IV >= {IV_THRESHOLD_DROP}): "
             f"{len(features_to_keep)}")
    log.info(f"  Features to drop: "
             f"{len(iv_table) - len(features_to_keep)}")

    return df_train_woe, df_test_woe, iv_table, binning_process


def pd_to_score(pd_values: np.ndarray,
                pdo: int = 20,
                base_score: int = 600,
                base_odds: float = 50.0) -> np.ndarray:
    """
    Convert probability of default to credit score (PDO calibration).

    The PDO (Points to Double the Odds) formula converts log-odds output
    from a logistic regression into a familiar 300-850 credit score scale.

    Args:
        pd_values: Array of PD estimates (0-1).
        pdo: Points needed to double the odds. Standard = 20.
        base_score: Score at base_odds. Standard = 600.
        base_odds: Good:Bad ratio at base_score. Standard = 50.

    Returns:
        Integer credit scores clipped to 300-850.
    """
    pd_values = np.clip(pd_values, 1e-6, 1 - 1e-6)
    odds   = (1 - pd_values) / pd_values
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    scores = offset + factor * np.log(odds)
    return np.clip(np.round(scores).astype(int), 300, 850)


def assign_risk_tier(scores: np.ndarray) -> np.ndarray:
    """
    Assign A-E risk tier labels based on credit scores.

    Tier thresholds (from config.py):
      A: 720-850  Very low risk
      B: 680-719  Low risk
      C: 630-679  Moderate risk — grey zone
      D: 580-629  Elevated risk
      E: 300-579  High risk
    """
    tiers = np.where(scores >= 720, "A",
            np.where(scores >= 680, "B",
            np.where(scores >= 630, "C",
            np.where(scores >= 580, "D", "E"))))
    return tiers
