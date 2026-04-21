"""
src/models/lgd_ead_model.py
────────────────────────────
Loss Given Default (LGD) and Exposure at Default (EAD) models.

Together with PD, these complete the Basel III Expected Loss trinity:
  Expected Loss = PD × LGD × EAD

LGD MODEL
─────────
LGD = fraction of the outstanding balance lost when a borrower defaults,
after recovery through collections and asset liquidation.

Key insight: LGD differs fundamentally by product type.

  Unsecured (personal loans):
    No collateral. Recovery comes only from collections.
    Typical LGD: 60-80%. Lender loses most of the balance.

  Secured (HELOC):
    Collateral = home equity. Lender can foreclose and sell.
    LGD driven primarily by LTV:
      LTV 60% → LGD ~15% (large equity cushion, easy recovery)
      LTV 85% → LGD ~40% (thin cushion, recovery costs bite)

Model approach: Gradient Boosting Regression on the defaulted
loan subset. Target = net_loss / original_balance.

EAD MODEL
─────────
EAD = total outstanding balance at the moment of default.

  Term loans (LendingClub personal loans):
    EAD follows the amortization schedule. Each payment reduces
    principal, so EAD = remaining_balance = calculable from
    loan_amount, term, rate, and time-on-book.
    We model EAD as a fraction of original loan amount.

  Revolving credit (HELOC):
    Borrowers often draw down additional amounts before defaulting.
    EAD = committed_limit × Credit Conversion Factor (CCF).
    CCF estimated from observed drawdown patterns on defaulted HELOCs.
    Typical CCF for HELOCs: 0.85-0.95 (borrowers draw most of the limit).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import logging

log = logging.getLogger(__name__)


# ── LGD Model ─────────────────────────────────────────────────────

def prepare_lgd_data(df: pd.DataFrame) -> tuple:
    """
    Prepare training data for the LGD model.

    Filters to defaulted loans only, computes LGD target as
    net_loss / loan_amount. For synthetic data, LGD is engineered
    based on product type and LTV (documented assumptions).

    Args:
        df: Full harmonized DataFrame.

    Returns:
        (X_lgd, y_lgd) feature matrix and LGD target.
    """
    # Filter to defaulted loans
    defaults = df[df["default_flag"] == 1].copy()
    log.info(f"  Defaulted loans for LGD modelling: {len(defaults):,}")

    # ── LGD Target Engineering ────────────────────────────────────
    # On real LendingClub data: LGD = (loan_amnt - recoveries) / loan_amnt
    # On synthetic data: we engineer a realistic LGD proxy

    rng = np.random.default_rng(42)
    n   = len(defaults)

    # Base LGD by product type
    # Unsecured: LGD ~ Beta(4, 2) centred around 0.65
    # Secured: LGD driven by LTV — lower LTV = lower LGD
    lgd_unsecured = rng.beta(4, 2, n).clip(0.3, 0.95)
    lgd_secured   = (defaults["ltv_ratio"].fillna(0.65) * 0.55 +
                     rng.normal(0, 0.08, n)).clip(0.05, 0.85)

    defaults["lgd"] = np.where(
        defaults["product_type"] == 1,
        lgd_secured,
        lgd_unsecured
    )

    # Features for LGD model
    lgd_features = [
        "ltv_ratio",
        "loan_amount",
        "credit_score",
        "num_derogatory_marks",
        "dti",
        "product_type",
        "months_since_oldest_trade",
        "total_accounts",
    ]
    available = [f for f in lgd_features if f in defaults.columns]

    X_lgd = defaults[available].fillna(0)
    y_lgd = defaults["lgd"]

    log.info(f"  LGD target: mean={y_lgd.mean():.3f}  "
             f"std={y_lgd.std():.3f}  "
             f"range=[{y_lgd.min():.3f}, {y_lgd.max():.3f}]")
    log.info(f"  Unsecured mean LGD: "
             f"{defaults[defaults['product_type']==0]['lgd'].mean():.3f}")
    log.info(f"  Secured mean LGD:   "
             f"{defaults[defaults['product_type']==1]['lgd'].mean():.3f}")

    return X_lgd, y_lgd, available


def train_lgd_model(X_lgd: pd.DataFrame,
                    y_lgd: pd.Series) -> GradientBoostingRegressor:
    """
    Train gradient boosting LGD regression model.

    Args:
        X_lgd: Feature matrix (defaulted loans only).
        y_lgd: LGD target (0-1 fraction of loan lost).

    Returns:
        Fitted GradientBoostingRegressor.
    """
    log.info("Training LGD model (GradientBoostingRegressor) ...")

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        loss="squared_error"
    )
    model.fit(X_lgd, y_lgd)

    # In-sample metrics
    y_pred = model.predict(X_lgd).clip(0, 1)
    rmse   = np.sqrt(mean_squared_error(y_lgd, y_pred))
    r2     = r2_score(y_lgd, y_pred)

    log.info(f"  LGD model — RMSE: {rmse:.4f}  R²: {r2:.4f}")
    log.info("  Note: LGD models typically show low R² (0.10-0.25 is normal)")
    log.info("  Reason: recovery rates have high inherent uncertainty")

    return model


def predict_lgd(model,
                df: pd.DataFrame,
                lgd_features: list) -> np.ndarray:
    """
    Predict LGD for all loans (not just defaulted).

    Args:
        model: Fitted LGD model.
        df: DataFrame to score.
        lgd_features: Feature columns used by LGD model.

    Returns:
        Array of LGD predictions (0-1).
    """
    available = [f for f in lgd_features if f in df.columns]
    X = df[available].fillna(0)
    lgd_pred = model.predict(X).clip(0, 1)
    return lgd_pred


# ── EAD Model ─────────────────────────────────────────────────────

def compute_ead_term_loan(loan_amount: np.ndarray,
                           months_on_book: np.ndarray = None,
                           loan_term_months: np.ndarray = None,
                           interest_rate: np.ndarray = None) -> np.ndarray:
    """
    Compute EAD for term loans using amortization schedule.

    For term loans, EAD = remaining principal balance.
    As payments are made, the balance declines predictably.

    Simple approach: EAD fraction = 1 - (months_on_book / loan_term)
    This assumes linear amortization (conservative approximation).

    Args:
        loan_amount: Original loan amounts.
        months_on_book: Months since origination (if unknown, use midpoint).
        loan_term_months: Original loan term.
        interest_rate: Annual interest rate (not used in simple approach).

    Returns:
        Array of EAD values.
    """
    if months_on_book is None:
        # If time on book unknown, assume midpoint of loan term
        # Conservative: more outstanding balance = higher exposure
        if loan_term_months is not None:
            months_on_book = loan_term_months * 0.40
        else:
            months_on_book = np.full(len(loan_amount), 18.0)  # 18 months default

    if loan_term_months is None:
        loan_term_months = np.full(len(loan_amount), 36.0)

    # EAD fraction = proportion of original balance remaining
    ead_fraction = 1.0 - (months_on_book / loan_term_months)
    ead_fraction = np.clip(ead_fraction, 0.05, 1.0)

    return loan_amount * ead_fraction


def compute_ead_revolving(loan_amount: np.ndarray,
                           product_type: np.ndarray,
                           ccf_heloc: float = 0.90) -> np.ndarray:
    """
    Compute EAD for revolving credit (HELOC) using Credit Conversion Factor.

    CCF accounts for the fact that revolving borrowers typically draw
    down their credit line further before defaulting.

    Industry typical CCF for HELOCs: 0.85-0.95
    We use 0.90 as a conservative middle estimate.

    Args:
        loan_amount: Credit limit (committed amount).
        product_type: 0=unsecured, 1=secured/revolving.
        ccf_heloc: Credit conversion factor for HELOC.

    Returns:
        Array of EAD values.
    """
    # For revolving (HELOC): EAD = limit × CCF
    # For term loans: use the standard amortization approach
    ead = np.where(
        product_type == 1,
        loan_amount * ccf_heloc,
        loan_amount  # Term loans: EAD = original amount (simplified)
    )
    return ead


def compute_expected_loss(pd_values: np.ndarray,
                           lgd_values: np.ndarray,
                           ead_values: np.ndarray) -> np.ndarray:
    """
    Compute Expected Loss = PD × LGD × EAD.

    This is the fundamental Basel III credit risk formula.
    EL represents the expected monetary loss on each loan.

    Args:
        pd_values:  PD estimates (0-1).
        lgd_values: LGD estimates (0-1).
        ead_values: EAD in currency units.

    Returns:
        Expected loss per loan in currency units.
    """
    return pd_values * lgd_values * ead_values


def summarise_el_portfolio(df: pd.DataFrame,
                            pd_col: str = "pd_score",
                            lgd_col: str = "lgd_estimate",
                            ead_col: str = "ead_estimate",
                            el_col: str = "expected_loss"
                            ) -> pd.DataFrame:
    """
    Portfolio-level Expected Loss summary by product and risk tier.

    Args:
        df: DataFrame with PD, LGD, EAD, and EL columns.

    Returns:
        Summary DataFrame.
    """
    if el_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for pt, ptname in [(0, "Unsecured"), (1, "Secured"), (-1, "Total")]:
        if pt == -1:
            subset = df
        else:
            subset = df[df["product_type"] == pt]

        if len(subset) == 0:
            continue

        rows.append({
            "product":        ptname,
            "n_loans":        len(subset),
            "total_ead":      subset[ead_col].sum(),
            "avg_pd":         subset[pd_col].mean(),
            "avg_lgd":        subset[lgd_col].mean(),
            "total_el":       subset[el_col].sum(),
            "el_rate":        subset[el_col].sum() / subset[ead_col].sum()
                              if subset[ead_col].sum() > 0 else 0,
        })

    return pd.DataFrame(rows)
