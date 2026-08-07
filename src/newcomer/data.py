"""
src/newcomer/data.py
────────────────────
STEP 4 (data) — synthetic newcomer population generator + real-file loader.

Follows the same philosophy as the platform's src/data/sample_generator.py:
generate an honest synthetic population so everything runs out of the box, and
swap in a real file the moment one is supplied.

WHAT MAKES THE SYNTHETIC DATA HONEST
────────────────────────────────────
The default outcome (repaid / default) is NOT random. It is generated to depend
on the applicant's own alternative-payment behaviour and affordability (DTI):
weak payers and over-stretched borrowers default more. That is what makes the
data learnable — a model can recover the relationship — while the OVERALL
default rate is calibrated to config.TARGET_DEFAULT_RATE (the 12% dial).

Every knob used here lives in config.py, in plain sight.

REAL-FILE SWAP
──────────────
`load_newcomers(path)` reads a CSV with the same columns. `get_newcomer_data()`
returns the real file if present, else the synthetic population — exactly the
"real data files found → use them, else synthetic fallback" pattern build.py
already uses for the core platform.
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.newcomer import config
from src.newcomer.alt_payment import compute_alt_payment_score, STREAMS
from src.newcomer.policy import compute_newcomer_dti

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _draw_reliability(rng: np.random.Generator) -> float:
    """
    A person's underlying on-time tendency. A mixture so the population has
    mostly-reliable payers plus a realistic minority of weaker ones — otherwise
    nobody would ever be declined on payment history.
    """
    g = rng.random()
    if g < 0.70:
        return float(rng.beta(9, 1))     # strong  (mean ~0.90)
    if g < 0.90:
        return float(rng.beta(4, 2))     # moderate (mean ~0.67)
    return float(rng.beta(2, 3))         # weak    (mean ~0.40)


def generate_newcomers(n: int = 2000, random_state: int = 42) -> list[dict]:
    """
    Generate `n` synthetic newcomer applicant records (list of dicts), each with
    the fields the scorer and policy expect, plus a calibrated `default_flag`.
    """
    log.info("Generating %s synthetic newcomer records ...", f"{n:,}")
    rng = np.random.default_rng(random_state)
    records: list[dict] = []

    for i in range(n):
        months = int(rng.integers(config.MONTHS_IN_COUNTRY_MIN,
                                  config.MONTHS_IN_COUNTRY_MAX + 1))
        no_file = rng.random() < config.PCT_NO_FILE
        reliability = _draw_reliability(rng)

        income = float(np.clip(rng.lognormal(np.log(50_000), 0.35), 24_000, 95_000))
        monthly_income = income / 12.0
        rent = float(np.clip(rng.normal(0.30, 0.05) * monthly_income, 900, 3_200))
        utilities = float(rng.uniform(100, 200))
        phone = float(rng.uniform(40, 80))
        has_insurance = rng.random() < 0.60
        insurance = float(rng.uniform(80, 160)) if has_insurance else 0.0

        def reported(delay_max: int, present: bool = True) -> int:
            if not present:
                return 0
            delay = int(rng.integers(0, delay_max + 1))   # account opened after arrival
            return max(0, months - delay)

        def on_time(rep: int) -> int:
            if rep <= 0:
                return 0
            sr = float(np.clip(reliability + rng.normal(0, 0.05), 0.0, 1.0))
            return int(round(rep * sr))

        rent_rep = reported(1)
        util_rep = reported(2)
        phone_rep = reported(1)
        ins_rep = reported(3, present=has_insurance)

        records.append({
            "id": f"NCG-{i:05d}",
            "newcomer_flag": True,
            "months_in_country": months,
            "bureau_status": "no_file" if no_file else "thin",
            "credit_score": None if no_file else int(rng.integers(520, 661)),
            "rent_on_time": on_time(rent_rep), "rent_reported": rent_rep,
            "rent_monthly": round(rent, 2),
            "utilities_on_time": on_time(util_rep), "utilities_reported": util_rep,
            "utilities_monthly": round(utilities, 2),
            "phone_on_time": on_time(phone_rep), "phone_reported": phone_rep,
            "phone_monthly": round(phone, 2),
            "insurance_on_time": on_time(ins_rep), "insurance_reported": ins_rep,
            "insurance_monthly": round(insurance, 2),
            "annual_income": round(income, 2),
            "loan_amount": round(float(np.clip(rng.lognormal(np.log(9_000), 0.5),
                                               3_000, 20_000)), 2),
            "loan_term_months": int(rng.choice([24, 36, 60])),
        })

    _assign_outcomes(records, rng)
    n_bad = sum(r["default_flag"] for r in records)
    log.info("  Default rate: %.1f%% (target %.0f%%)",
             100 * n_bad / max(len(records), 1),
             100 * config.TARGET_DEFAULT_RATE)
    return records


def _assign_outcomes(records: list[dict], rng: np.random.Generator) -> None:
    """
    Assign `default_flag` so risk tracks payment behaviour + affordability, with
    the overall rate calibrated to config.TARGET_DEFAULT_RATE via a logistic
    intercept found by bisection.
    """
    risk = np.empty(len(records))
    for j, r in enumerate(records):
        sig = compute_alt_payment_score(r)
        score = 50.0 if sig["alt_payment_score"] is None else sig["alt_payment_score"]
        dti = min(compute_newcomer_dti(r), 100.0)
        # Worse payments (low score) and higher DTI → higher risk.
        risk[j] = 0.65 * (1.0 - score / 100.0) + 0.35 * (dti / 80.0)

    z = (risk - risk.mean()) / (risk.std() + 1e-9)
    k = 1.3                                   # spread of risk → probability
    target = config.TARGET_DEFAULT_RATE
    lo, hi = -12.0, 12.0                       # bisection on the intercept
    for _ in range(60):
        c = (lo + hi) / 2.0
        if _sigmoid(c + k * z).mean() > target:
            hi = c
        else:
            lo = c
    probs = _sigmoid(c + k * z)
    draws = rng.random(len(records))
    for r, p, d in zip(records, probs, draws):
        r["default_flag"] = int(d < p)


# ── Real-file swap ───────────────────────────────────────────────────

def to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert a list of applicant records to a DataFrame."""
    return pd.DataFrame(records)


def load_newcomers(path: str | Path) -> list[dict]:
    """Load a real newcomer file (CSV) with the same columns as the generator."""
    df = pd.read_csv(path)
    log.info("Loaded %s real newcomer rows from %s", f"{len(df):,}", path)
    return df.to_dict(orient="records")


def get_newcomer_data(path: str | Path | None = None,
                      n: int = 2000,
                      random_state: int = 42) -> list[dict]:
    """
    Real file if `path` is given and exists, else the synthetic population.
    Mirrors the core platform's real-else-synthetic behaviour.
    """
    if path is not None and Path(path).exists():
        return load_newcomers(path)
    if path is not None:
        log.warning("Newcomer file %s not found — using synthetic fallback", path)
    return generate_newcomers(n=n, random_state=random_state)


if __name__ == "__main__":
    recs = generate_newcomers(1000)
    df = to_dataframe(recs)
    print(df[["months_in_country", "bureau_status", "annual_income",
              "default_flag"]].describe(include="all"))
