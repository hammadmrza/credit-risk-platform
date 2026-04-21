"""
src/models/vintage_analysis.py
────────────────────────────────
Vintage analysis and model monitoring utilities.

VINTAGE ANALYSIS
────────────────
Tracks cohorts of loans originated in the same period and plots
their cumulative default rates as they age on the portfolio.

Used by:
  - Portfolio managers (monthly risk review)
  - Model validators (out-of-time performance)
  - Regulators (evidence of model stability)

PSI / CSI MONITORING
─────────────────────
PSI (Population Stability Index): measures score distribution drift.
CSI (Characteristic Stability Index): measures feature distribution drift.

OSFI E-23 triggers:
  PSI < 0.10  → No action required
  PSI 0.10-0.25 → Investigate
  PSI > 0.25  → Mandatory model review / potential redevelopment
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

log = logging.getLogger(__name__)


def compute_vintage_curves(df: pd.DataFrame,
                            vintage_col: str = "origination_year",
                            target: str = "default_flag"
                            ) -> pd.DataFrame:
    """
    Compute default rates by origination cohort.

    Args:
        df: DataFrame with vintage and target columns.
        vintage_col: Column defining the cohort (year or quarter).
        target: Binary default flag.

    Returns:
        DataFrame with vintage, n_loans, default_rate, rolling_avg.
    """
    curves = (df.groupby(vintage_col)[target]
              .agg(n_loans="count", default_rate="mean")
              .reset_index()
              .sort_values(vintage_col))

    curves["rolling_3_avg"] = (curves["default_rate"]
                                .rolling(3, min_periods=1).mean())
    curves["vs_overall_avg"] = (curves["default_rate"] -
                                 curves["default_rate"].mean())
    return curves


def compute_psi(expected: np.ndarray,
                actual: np.ndarray,
                n_buckets: int = 10) -> float:
    """
    Population Stability Index.

    Compares the distribution of model scores between:
      expected = development sample (when model was built)
      actual   = monitoring sample (current production)

    Returns:
        PSI value. < 0.10 = stable, 0.10-0.25 = watch, > 0.25 = alert.
    """
    breakpoints = np.percentile(expected,
                                 np.linspace(0, 100, n_buckets + 1))
    breakpoints = np.unique(breakpoints)

    def bucket_pct(arr, breaks):
        counts, _ = np.histogram(arr, bins=breaks)
        pct = counts / len(arr)
        return np.where(pct == 0, 1e-6, pct)

    exp_pct = bucket_pct(expected, breakpoints)
    act_pct = bucket_pct(actual, breakpoints)

    # Align lengths
    min_len = min(len(exp_pct), len(act_pct))
    exp_pct = exp_pct[:min_len]
    act_pct = act_pct[:min_len]

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def compute_csi(df_dev: pd.DataFrame,
                df_mon: pd.DataFrame,
                features: list,
                n_buckets: int = 10) -> pd.DataFrame:
    """
    Characteristic Stability Index for each feature.

    Args:
        df_dev: Development sample DataFrame.
        df_mon: Monitoring sample DataFrame.
        features: Features to monitor.
        n_buckets: Number of bins for PSI calculation.

    Returns:
        DataFrame with feature, CSI value, and stability label.
    """
    rows = []
    for feat in features:
        if feat not in df_dev.columns or feat not in df_mon.columns:
            continue
        try:
            dev_vals = df_dev[feat].dropna().values
            mon_vals = df_mon[feat].dropna().values
            if len(dev_vals) < 50 or len(mon_vals) < 50:
                continue
            csi = compute_psi(dev_vals, mon_vals, n_buckets)
            rows.append({
                "feature": feat,
                "csi":     round(csi, 4),
                "status":  ("OK" if csi < 0.10
                            else "WATCH" if csi < 0.25
                            else "ALERT — investigate")
            })
        except Exception:
            pass

    result = pd.DataFrame(rows).sort_values("csi", ascending=False)
    return result


def psi_label(psi: float) -> str:
    """Human-readable PSI interpretation."""
    if psi < 0.10:
        return "Stable — no action required"
    elif psi < 0.25:
        return "Moderate shift — investigate"
    else:
        return "Significant shift — model review required (OSFI E-23)"
