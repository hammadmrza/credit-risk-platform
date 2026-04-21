"""
src/data/eda_utils.py
─────────────────────
Reusable EDA and analysis utilities used across Phase 2 notebooks.

Functions cover:
- Default rate analysis by segment
- Feature distribution summaries
- Missing data audit
- Class imbalance reporting
- Vintage analysis helpers
- Correlation and univariate stats
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")


# ── Default rate analysis ────────────────────────────────────────

def default_rate_by_feature(df: pd.DataFrame,
                             feature: str,
                             target: str = "default_flag",
                             min_obs: int = 50) -> pd.DataFrame:
    """
    Default rate and volume for each category/bin of a feature.

    Args:
        df: DataFrame with feature and target columns.
        feature: Column name to group by.
        target: Binary target column (default = 'default_flag').
        min_obs: Minimum observations per group to include.

    Returns:
        DataFrame with columns: feature_value, n_loans, n_defaults,
        default_rate, pct_of_portfolio.
    """
    result = (df.groupby(feature)[target]
              .agg(n_loans="count", n_defaults="sum")
              .reset_index())
    result["default_rate"]      = result["n_defaults"] / result["n_loans"]
    result["pct_of_portfolio"]  = result["n_loans"] / result["n_loans"].sum()
    result = result[result["n_loans"] >= min_obs].copy()
    result = result.rename(columns={feature: "feature_value"})
    result["feature"] = feature
    result["default_rate_fmt"] = result["default_rate"].map("{:.2%}".format)
    return result.sort_values("default_rate", ascending=False)


def default_rate_by_score_band(df: pd.DataFrame,
                                score_col: str = "credit_score",
                                target: str = "default_flag",
                                bins: int = 10) -> pd.DataFrame:
    """
    Default rate across equal-width score bands.

    Args:
        df: DataFrame.
        score_col: Numeric score column.
        target: Binary target.
        bins: Number of bands.

    Returns:
        DataFrame with band, n_loans, default_rate.
    """
    df = df.copy()
    df["score_band"] = pd.cut(df[score_col], bins=bins)
    result = (df.groupby("score_band")[target]
              .agg(n_loans="count", n_defaults="sum")
              .reset_index())
    result["default_rate"] = result["n_defaults"] / result["n_loans"]
    return result


def segment_comparison(df: pd.DataFrame,
                        segment_col: str = "product_type",
                        target: str = "default_flag",
                        features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare key statistics between segments (e.g. secured vs unsecured).

    Args:
        df: DataFrame.
        segment_col: Column defining segments.
        target: Binary target.
        features: Numeric features to summarize. If None, auto-detects.

    Returns:
        DataFrame with one row per feature and one column per segment.
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features
                    if f not in [target, segment_col, "origination_year"]]

    segments = df[segment_col].unique()
    rows = []
    for feat in features:
        row = {"feature": feat}
        for seg in sorted(segments):
            seg_data = df[df[segment_col] == seg][feat].dropna()
            row[f"seg_{seg}_mean"]   = seg_data.mean()
            row[f"seg_{seg}_median"] = seg_data.median()
            row[f"seg_{seg}_null%"]  = df[df[segment_col] == seg][feat].isna().mean() * 100
        rows.append(row)

    result = pd.DataFrame(rows)
    return result


# ── Missing data audit ───────────────────────────────────────────

def missing_data_report(df: pd.DataFrame,
                         threshold_warn: float = 0.05,
                         threshold_drop: float = 0.40) -> pd.DataFrame:
    """
    Comprehensive missing data audit with action recommendations.

    Args:
        df: DataFrame to audit.
        threshold_warn: Null rate above which to flag for attention.
        threshold_drop: Null rate above which to recommend dropping feature.

    Returns:
        DataFrame sorted by null rate descending.
    """
    null_counts = df.isnull().sum()
    null_rates  = null_counts / len(df)

    report = pd.DataFrame({
        "column":     null_counts.index,
        "null_count": null_counts.values,
        "null_rate":  null_rates.values,
        "dtype":      [str(df[c].dtype) for c in null_counts.index],
    })

    def recommend(rate):
        if rate == 0:
            return "OK"
        elif rate < threshold_warn:
            return "OK - impute"
        elif rate < threshold_drop:
            return "WARN - impute carefully"
        else:
            return "DROP or flag"

    report["recommendation"] = report["null_rate"].map(recommend)
    report["null_rate_fmt"]  = report["null_rate"].map("{:.2%}".format)
    return (report[report["null_count"] > 0]
            .sort_values("null_rate", ascending=False)
            .reset_index(drop=True))


# ── Class imbalance ──────────────────────────────────────────────

def imbalance_report(df: pd.DataFrame,
                      target: str = "default_flag",
                      segment_col: Optional[str] = "product_type"
                      ) -> Dict:
    """
    Class imbalance summary with SMOTE recommendation.

    Returns dict with overall and per-segment metrics.
    """
    overall_rate = df[target].mean()
    ratio        = (1 - overall_rate) / overall_rate

    report = {
        "overall_default_rate": overall_rate,
        "good_to_bad_ratio":    ratio,
        "smote_recommended":    ratio > 10,
        "segments": {}
    }

    if segment_col and segment_col in df.columns:
        for seg in sorted(df[segment_col].unique()):
            seg_data = df[df[segment_col] == seg]
            seg_rate = seg_data[target].mean()
            seg_ratio = (1 - seg_rate) / seg_rate if seg_rate > 0 else np.inf
            report["segments"][seg] = {
                "n": len(seg_data),
                "default_rate": seg_rate,
                "ratio": seg_ratio,
                "smote_recommended": seg_ratio > 10
            }

    return report


# ── Vintage analysis ─────────────────────────────────────────────

def compute_vintage_curves(df: pd.DataFrame,
                            vintage_col: str = "origination_quarter",
                            target: str = "default_flag"
                            ) -> pd.DataFrame:
    """
    Compute vintage default rates for cohort analysis.

    Since LendingClub data has final outcomes (not monthly snapshots),
    this shows the terminal default rate per origination cohort.

    Args:
        df: DataFrame with vintage and target columns.
        vintage_col: Column defining cohort (e.g. origination_quarter).
        target: Binary default flag.

    Returns:
        DataFrame with vintage, n_loans, default_rate, sorted by vintage.
    """
    curves = (df.groupby(vintage_col)[target]
              .agg(n_loans="count", default_rate="mean")
              .reset_index()
              .sort_values(vintage_col))

    curves["vintage_num"] = range(len(curves))
    curves["3q_rolling_avg"] = (curves["default_rate"]
                                 .rolling(3, min_periods=1)
                                 .mean())
    return curves


def compute_vintage_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Vintage curves split by product_type."""
    results = []
    for pt, name in [(0, "Unsecured"), (1, "Secured")]:
        subset = df[df["product_type"] == pt]
        curves = compute_vintage_curves(subset)
        curves["product_name"] = name
        results.append(curves)
    return pd.concat(results, ignore_index=True)


# ── Correlation analysis ─────────────────────────────────────────

def feature_target_correlations(df: pd.DataFrame,
                                  features: List[str],
                                  target: str = "default_flag"
                                  ) -> pd.DataFrame:
    """
    Point-biserial correlations between numeric features and binary target.
    Sorted by absolute correlation descending.

    Args:
        df: DataFrame.
        features: Numeric feature columns.
        target: Binary target column.

    Returns:
        DataFrame with feature, correlation, abs_correlation, direction.
    """
    corrs = []
    for f in features:
        if f in df.columns and f != target:
            try:
                c = df[[f, target]].dropna()[f].corr(
                    df[[f, target]].dropna()[target]
                )
                corrs.append({
                    "feature":         f,
                    "correlation":     round(c, 4),
                    "abs_correlation": abs(round(c, 4)),
                    "direction":       "positive" if c > 0 else "negative"
                })
            except Exception:
                pass

    result = (pd.DataFrame(corrs)
              .sort_values("abs_correlation", ascending=False)
              .reset_index(drop=True))
    return result


# ── ADS lift analysis ────────────────────────────────────────────

def ads_lift_analysis(df: pd.DataFrame,
                       ads_col: str = "alt_data_score",
                       target: str = "default_flag",
                       thin_file_col: str = "thin_file_flag"
                       ) -> pd.DataFrame:
    """
    Measure ADS lift over traditional features, especially for thin-file.

    Returns:
        DataFrame with AUC comparison (traditional vs traditional+ADS)
        for full portfolio and thin-file segment.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    traditional_features = [
        "credit_score", "dti", "credit_utilization",
        "num_inquiries_last_6m", "num_derogatory_marks",
        "months_since_recent_delinquency", "total_accounts"
    ]

    available = [f for f in traditional_features if f in df.columns]
    results   = []

    for segment_name, segment_df in [
        ("Full portfolio", df),
        ("Thin-file only", df[df[thin_file_col] == True] if thin_file_col in df.columns else df.head(0))
    ]:
        if len(segment_df) < 100:
            continue

        work = segment_df[available + [ads_col, target]].dropna()
        if len(work) < 50:
            continue

        y = work[target]

        # Traditional features only
        pipe_trad = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=200, random_state=42))
        ])
        try:
            pipe_trad.fit(work[available], y)
            auc_trad = roc_auc_score(y, pipe_trad.predict_proba(work[available])[:, 1])
        except Exception:
            auc_trad = np.nan

        # Traditional + ADS
        pipe_ads = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=200, random_state=42))
        ])
        try:
            pipe_ads.fit(work[available + [ads_col]], y)
            auc_ads = roc_auc_score(
                y, pipe_ads.predict_proba(work[available + [ads_col]])[:, 1]
            )
        except Exception:
            auc_ads = np.nan

        results.append({
            "segment":       segment_name,
            "n":             len(work),
            "default_rate":  y.mean(),
            "auc_traditional": round(auc_trad, 4) if not np.isnan(auc_trad) else None,
            "auc_with_ads":    round(auc_ads, 4) if not np.isnan(auc_ads) else None,
            "auc_lift":        round(auc_ads - auc_trad, 4)
                               if not (np.isnan(auc_trad) or np.isnan(auc_ads))
                               else None
        })

    return pd.DataFrame(results)


# ── Summary statistics ───────────────────────────────────────────

def feature_summary(df: pd.DataFrame,
                     features: List[str],
                     target: str = "default_flag") -> pd.DataFrame:
    """
    Comprehensive summary statistics for a list of features.
    Includes mean/median/std/null rate and correlation with target.
    """
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        col = df[f]
        corr = col.corr(df[target]) if f != target else 1.0
        rows.append({
            "feature":  f,
            "mean":     round(col.mean(), 3),
            "median":   round(col.median(), 3),
            "std":      round(col.std(), 3),
            "min":      round(col.min(), 3),
            "max":      round(col.max(), 3),
            "null_%":   round(col.isna().mean() * 100, 1),
            "corr_target": round(corr, 4)
        })
    return pd.DataFrame(rows)
