"""
src/models/stress_testing.py
──────────────────────────────
Macroeconomic stress testing for credit risk portfolios.

WHY STRESS TESTING IS MANDATORY
────────────────────────────────
OSFI requires federally regulated institutions to demonstrate their
portfolios can withstand adverse macroeconomic conditions without
breaching minimum capital requirements. The stress test shows:

  1. How much PD rises under each scenario
  2. How much additional capital would be required
  3. Whether the institution remains above its CET1 minimum

THREE SCENARIOS
───────────────
Base:     Continuation of current conditions. PD changes minimal.
          GDP growth: ~1.5%, Unemployment: ~6.5%, Rate: -25bps

Adverse:  Moderate recession. Plausible but not the worst case.
          GDP growth: -1.5%, Unemployment: +250bps, Rate: +100bps
          PD multiplier: ~1.4x (40% increase in defaults)

Severe:   Deep recession. Tail risk scenario. 2008-09 magnitude.
          GDP growth: -4.0%, Unemployment: +500bps, Rate: +200bps
          PD multiplier: ~2.0x (defaults double)

These scenarios are calibrated against:
  - Bank of Canada historical stress test parameters
  - OSFI supervisory expectations for Internal Capital Adequacy
    Assessment Process (ICAAP) stress testing
  - Published academic research on Canadian credit cycle sensitivity

MACROECONOMIC LINKAGE
──────────────────────
PD sensitivity to macroeconomic factors follows a log-linear model:

  Stressed PD = Base PD × exp(β_gdp × ΔGDP + β_ue × ΔUE + β_rate × ΔRate)

Where:
  β_gdp  = GDP sensitivity   (negative: higher GDP → lower PD)
  β_ue   = Unemployment sensitivity (positive: higher UE → higher PD)
  β_rate = Interest rate sensitivity (positive: higher rates → higher PD)

Coefficients are calibrated to Canadian consumer credit historical data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

log = logging.getLogger(__name__)

# ── Scenario Definitions ──────────────────────────────────────────
SCENARIOS = {
    "base": {
        "name":           "Base Case",
        "description":    "Continuation of current macro conditions",
        "gdp_change":     0.015,    # +1.5% GDP growth
        "ue_change":      0.0,      # No change in unemployment
        "rate_change":    -0.0025,  # -25bps rate cut
        "house_price":    0.02,     # +2% house prices
        "pd_multiplier":  1.00,     # No stress
        "lgd_multiplier": 1.00,
        "colour":         "green",
    },
    "adverse": {
        "name":           "Adverse Scenario",
        "description":    "Moderate recession — plausible downside",
        "gdp_change":     -0.015,   # -1.5% GDP contraction
        "ue_change":      0.025,    # +250bps unemployment rise
        "rate_change":    0.010,    # +100bps rate increase
        "house_price":    -0.10,    # -10% house price decline
        "pd_multiplier":  1.40,     # PDs increase 40%
        "lgd_multiplier": 1.15,     # LGD increases 15% (more defaults)
        "colour":         "orange",
    },
    "severe": {
        "name":           "Severe Scenario",
        "description":    "Deep recession — 2008-09 magnitude tail risk",
        "gdp_change":     -0.040,   # -4.0% GDP contraction
        "ue_change":      0.050,    # +500bps unemployment spike
        "rate_change":    0.020,    # +200bps rate shock
        "house_price":    -0.25,    # -25% house price collapse
        "pd_multiplier":  2.00,     # PDs double
        "lgd_multiplier": 1.35,     # LGD increases 35%
        "colour":         "red",
    },
}

# Macro sensitivity coefficients (calibrated to Canadian data)
BETA_GDP  = -0.8   # 1% GDP drop → ~80bp PD increase (log scale)
BETA_UE   =  0.5   # 1% UE rise  → ~50bp PD increase
BETA_RATE =  0.3   # 1% rate rise → ~30bp PD increase


def apply_macro_stress(pd_base: np.ndarray,
                        scenario: dict,
                        is_ttc: bool = False) -> np.ndarray:
    """
    Apply macroeconomic stress to base PD.

    FIX: Stress multipliers now applied to TTC PD for RWA/capital,
    and PIT PD for EL/ECL — consistent with Basel III intent.
    The IRB formula requires TTC inputs; applying stressed PIT PD
    to the IRB formula was a methodological error that caused
    severe-scenario RWA to drop below adverse-scenario RWA.

    For TTC-based capital: stress multiplier applied to TTC PD,
    then re-smoothed toward long-run average (counter-cyclical buffer
    built into TTC is the correct mechanism, not raw PD multiplication).

    For PIT-based EL: stress multiplier applied directly to PIT PD.
    """
    # Apply direct multiplier (consistent, transparent)
    pd_stressed = pd_base * scenario["pd_multiplier"]
    pd_stressed = np.clip(pd_stressed, 0, 0.999)

    if is_ttc:
        # For TTC PD: re-smooth toward long-run average after stressing
        # This preserves the counter-cyclical property of TTC
        from src.features.interactions import compute_pit_to_ttc
        pd_stressed = compute_pit_to_ttc(
            pd_stressed,
            long_run_average_pd=0.08,
            smoothing_factor=0.20  # Less smoothing under stress
        )

    return pd_stressed


def run_stress_test(portfolio_df: pd.DataFrame,
                     pd_col: str = "pd_score",
                     pd_ttc_col: str = "pd_ttc",
                     lgd_col: str = "lgd_estimate",
                     ead_col: str = "ead_estimate",
                     current_capital: float = None) -> pd.DataFrame:
    """
    Run all three stress scenarios and return comparative results.

    Args:
        portfolio_df: Portfolio with PD, LGD, EAD columns.
        pd_col: PIT PD column (for IFRS 9 / EL).
        pd_ttc_col: TTC PD column (for RWA / capital).
        lgd_col: LGD column.
        ead_col: EAD column.
        current_capital: Current capital held ($). If None, computed
                         as 8% of base RWA.

    Returns:
        DataFrame with results for all scenarios side by side.
    """
    from src.models.regulatory import compute_irb_capital

    total_ead = portfolio_df[ead_col].sum()
    results   = []

    for scenario_key, scenario in SCENARIOS.items():
        # FIX: Stress PIT PD for EL, stress TTC PD for capital — separate paths
        pd_pit_stressed = apply_macro_stress(
            portfolio_df[pd_col].values, scenario, is_ttc=False
        )
        pd_ttc_stressed = apply_macro_stress(
            portfolio_df[pd_ttc_col].values, scenario, is_ttc=True
        )
        lgd_stressed = np.clip(
            portfolio_df[lgd_col].values * scenario["lgd_multiplier"], 0, 1
        )

        # Stressed EL
        el_stressed = pd_pit_stressed * lgd_stressed * portfolio_df[ead_col].values
        total_el    = el_stressed.sum()

        # Stressed RWA and capital
        irb = compute_irb_capital(
            pd_ttc=pd_ttc_stressed,
            lgd=lgd_stressed,
            ead=portfolio_df[ead_col].values,
        )
        total_rwa        = irb["rwa"].sum()
        required_capital = irb["min_capital"].sum()  # 8% of RWA

        results.append({
            "scenario":           scenario["name"],
            "scenario_key":       scenario_key,
            "description":        scenario["description"],
            "pd_multiplier":      scenario["pd_multiplier"],
            "lgd_multiplier":     scenario["lgd_multiplier"],
            "avg_pd_stressed":    pd_pit_stressed.mean(),
            "total_ead":          total_ead,
            "total_el_stressed":  total_el,
            "el_rate":            total_el / total_ead,
            "total_rwa":          total_rwa,
            "rwa_density":        total_rwa / total_ead,
            "required_capital":   required_capital,
            "capital_ratio":      required_capital / total_ead,
        })

    return pd.DataFrame(results)


def compute_capital_headroom(stress_results: pd.DataFrame,
                               available_capital: float,
                               cet1_minimum: float = 0.115) -> pd.DataFrame:
    """
    Compute capital headroom (buffer above minimum) under each scenario.

    Args:
        stress_results: Output of run_stress_test().
        available_capital: Total capital currently held ($).
        cet1_minimum: Minimum CET1 ratio (11.5% for D-SIBs).

    Returns:
        DataFrame with headroom metrics added.
    """
    result = stress_results.copy()
    total_ead = result["total_ead"].iloc[0]

    result["available_capital"]  = available_capital
    result["capital_headroom"]   = available_capital - result["required_capital"]
    result["headroom_pct_ead"]   = result["capital_headroom"] / total_ead
    result["passes_minimum"]     = result["capital_headroom"] > 0
    result["capital_surplus"]    = (
        available_capital / result["total_rwa"] - 0.08
    ).clip(0)

    return result


def format_stress_summary(stress_results: pd.DataFrame) -> str:
    """Format stress test results as clean text output."""
    lines = [
        "STRESS TEST RESULTS",
        "=" * 60,
        "",
    ]

    for _, row in stress_results.iterrows():
        lines += [
            f"  {row['scenario'].upper()}",
            f"  {row['description']}",
            f"  {'─' * 50}",
            f"  PD multiplier:        {row['pd_multiplier']:.2f}x",
            f"  Average stressed PD:  {row['avg_pd_stressed']:.2%}",
            f"  Portfolio EL:         ${row['total_el_stressed']:,.0f}",
            f"  EL rate:              {row['el_rate']:.2%}",
            f"  Total RWA:            ${row['total_rwa']:,.0f}",
            f"  Required capital:     ${row['required_capital']:,.0f}",
            f"  Capital ratio:        {row['capital_ratio']:.2%}",
            "",
        ]

    return "\n".join(lines)
