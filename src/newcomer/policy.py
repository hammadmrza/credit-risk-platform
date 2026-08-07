"""
src/newcomer/policy.py
──────────────────────
STEP 2 — the newcomer adjudication rule (REFER-first).

WHAT PROBLEM THIS SOLVES
────────────────────────
The core platform declines anyone below a 500 bureau score at a hard gate,
BEFORE any model or alternative data is considered (src/app/utils.py,
HARD_POLICY["min_credit_score"], evaluate_policy_rules). A true newcomer with
no bureau file is therefore auto-declined before their on-time rent, phone,
utility, and insurance history is ever looked at.

This module replaces that single gate — for newcomers only — with a rule that:
  1. keeps the affordability check ON (DTI, TDS-style, includes rent + new loan),
  2. keeps the fraud gate ON (a stub seam here; wired to the real model later),
  3. requires a genuine alternative-payment history (enough streams + months),
  4. and then, on a strong alt-payment score, routes to REFER — a human analyst
     with the payment summary attached.

REFER-FIRST
───────────
This module never returns APPROVE. The most a newcomer can earn here is REFER,
because the platform has no real newcomer repayment history to auto-approve
against yet. Weak history or poor affordability still declines automatically.

DECISION CODES (mirror the core platform's vocabulary)
──────────────────────────────────────────────────────
    REFER            — qualified: hand to an analyst with the payment summary
    DECLINE_CREDIT   — alt-payment score below the REFER threshold
    DECLINE_POLICY   — affordability (DTI) or insufficient alternative data
    DECLINE_FRAUD    — fraud gate tripped (stub until the fraud model is wired)
"""

from __future__ import annotations
import logging

from src.newcomer import config
from src.newcomer.alt_payment import get_alt_payment_signal

log = logging.getLogger(__name__)


# ── Affordability ────────────────────────────────────────────────────

def _monthly_loan_payment(principal: float, apr: float, term_months: int) -> float:
    """Amortised monthly payment for the requested loan (0% APR → straight-line)."""
    principal = float(principal or 0.0)
    term_months = int(term_months or 0)
    if term_months <= 0:
        return 0.0
    r = apr / 12.0
    if r == 0.0:
        return principal / term_months
    factor = (1.0 + r) ** term_months
    return principal * r * factor / (factor - 1.0)


def compute_newcomer_dti(applicant: dict) -> float:
    """
    Newcomer DTI (%), TDS-style — INCLUDES rent and the new loan payment.

        DTI = (rent + utilities + phone + insurance + new-loan payment)
              ─────────────────────────────────────────────────────────  × 100
                               monthly gross income

    Dollar amounts come from the applicant's monthly bill fields
    (<stream>_monthly) rather than the bureau, because a no-file newcomer has
    no bureau-reported obligations. This is an "alt-DTI" and is labelled as
    such wherever it is surfaced.
    """
    annual_income = float(applicant.get("annual_income", 0) or 0)
    monthly_income = annual_income / 12.0
    if monthly_income <= 0:
        return float("inf")        # no income → infinitely unaffordable

    bills = sum(
        float(applicant.get(f"{s}_monthly", 0) or 0)
        for s in ("rent", "utilities", "phone", "insurance")
    )
    loan_payment = _monthly_loan_payment(
        applicant.get("loan_amount", 0),
        config.ASSUMED_APR,
        applicant.get("loan_term_months", 0),
    )
    return round(100.0 * (bills + loan_payment) / monthly_income, 1)


# ── Adjudication ─────────────────────────────────────────────────────

def _analyst_summary(applicant: dict, signal: dict, dti: float) -> str:
    """One-line payment summary an analyst sees on a REFER."""
    bits = [
        f"{applicant.get('months_in_country', '?')} mo in Canada",
        "no bureau file" if applicant.get("bureau_status") == "no_file"
        else f"thin file (score {applicant.get('credit_score', '—')})",
    ]
    for stream in ("rent", "utilities", "phone", "insurance"):
        s = signal["per_stream"].get(stream)
        if s:
            bits.append(f"{stream} {s['on_time']}/{s['reported']}")
    bits.append(f"alt-score {signal['alt_payment_score']}")
    bits.append(f"alt-DTI {dti:.0f}%")
    return " · ".join(bits)


def adjudicate_newcomer(applicant: dict,
                        alt_signal: dict | None = None,
                        fraud_result: dict | None = None) -> dict:
    """
    Adjudicate a newcomer applicant. REFER-first: never returns APPROVE.

    Args:
        applicant:    dict with bill-payment fields, monthly $ amounts,
                      annual_income, loan_amount, loan_term_months, and
                      bureau_status ("no_file" | "thin").
        alt_signal:   optional pre-computed signal from get_alt_payment_signal;
                      computed here if not supplied.
        fraud_result: optional {"fraud_score": float, "available": bool};
                      the fraud gate is skipped when not supplied (stub seam
                      until the platform's fraud model is wired in).

    Returns:
        dict: decision, reason, dti, alt_payment_score, alt_tradelines,
              months_observed, analyst_summary.
    """
    if alt_signal is None:
        alt_signal = get_alt_payment_signal(applicant)

    score = alt_signal.get("alt_payment_score")
    tradelines = alt_signal.get("alt_tradelines", 0)
    months = alt_signal.get("months_observed", 0)
    dti = compute_newcomer_dti(applicant)

    def result(decision: str, reason: str) -> dict:
        return {
            "decision": decision,
            "reason": reason,
            "dti": dti,
            "alt_payment_score": score,
            "alt_tradelines": tradelines,
            "months_observed": months,
            "analyst_summary": _analyst_summary(applicant, alt_signal, dti),
        }

    # Step 1 — Fraud gate (stub: only fires if a real fraud result is passed in)
    if fraud_result and fraud_result.get("available"):
        from_score = float(fraud_result.get("fraud_score", 0.0))
        if from_score > 0.65:      # mirrors core FRAUD_DECLINE_THRESH
            return result(
                "DECLINE_FRAUD",
                f"Fraud probability {from_score:.0%} exceeds the fraud gate "
                f"threshold. Newcomer + thin file is a synthetic-identity risk "
                f"profile — the fraud check is applied in full.",
            )

    # Step 2 — Affordability (kept ON; TDS-style DTI includes rent + new loan)
    if dti > config.MAX_DTI:
        return result(
            "DECLINE_POLICY",
            f"Alt-DTI {dti:.0f}% exceeds the {config.MAX_DTI:.0f}% affordability "
            f"ceiling. Rent, bills, and the new loan together take too much of "
            f"income — a strong payment record does not override affordability.",
        )

    # Step 3 — Sufficient alternative-payment history to assess at all
    if tradelines < config.MIN_ALT_TRADELINES or months < config.MIN_HISTORY_MONTHS:
        return result(
            "DECLINE_POLICY",
            f"Insufficient alternative-data to assess: {tradelines} payment "
            f"stream(s) over {months} month(s) (need "
            f"{config.MIN_ALT_TRADELINES}+ streams and "
            f"{config.MIN_HISTORY_MONTHS}+ months). No bureau file to fall back on.",
        )

    # Step 4 — Alt-payment credit threshold
    if score is None or score < config.ALT_SCORE_REFER_MIN:
        return result(
            "DECLINE_CREDIT",
            f"Alt-payment score {score} is below the REFER threshold of "
            f"{config.ALT_SCORE_REFER_MIN}. Payment history is too weak to "
            f"support the application.",
        )

    # Step 5 — Qualified → REFER (never auto-approve)
    return result(
        "REFER",
        f"No/thin bureau file but a strong verified payment record "
        f"(alt-score {score}, {tradelines} streams over {months} months) and "
        f"affordable (alt-DTI {dti:.0f}%). Routed to an analyst for the final "
        f"decision — the model does not auto-approve this segment.",
    )
