"""
src/newcomer/demo.py
────────────────────
Runs a handful of sample newcomer applicants through Step 1 (the alternative-
payment score) and Step 2 (the REFER-first adjudication rule), and prints the
result table.

These five applicants are hand-built to span the range (strong → weak). The
full random data generator (Step 4) is separate. Run:

    python -m src.newcomer.demo
"""

from __future__ import annotations

from src.newcomer.alt_payment import get_alt_payment_signal
from src.newcomer.policy import adjudicate_newcomer


# id, months in country, bureau, and — for each stream — (on_time, reported, $/mo)
SAMPLE_NEWCOMERS = [
    {
        "id": "NC-001", "months_in_country": 14, "bureau_status": "no_file",
        "credit_score": None,
        "rent_on_time": 14, "rent_reported": 14, "rent_monthly": 1400,
        "utilities_on_time": 11, "utilities_reported": 12, "utilities_monthly": 150,
        "phone_on_time": 12, "phone_reported": 12, "phone_monthly": 60,
        "insurance_on_time": 14, "insurance_reported": 14, "insurance_monthly": 120,
        "annual_income": 58000, "loan_amount": 10000, "loan_term_months": 36,
    },
    {
        "id": "NC-002", "months_in_country": 8, "bureau_status": "no_file",
        "credit_score": None,
        "rent_on_time": 8, "rent_reported": 8, "rent_monthly": 1200,
        "utilities_on_time": 6, "utilities_reported": 8, "utilities_monthly": 120,
        "phone_on_time": 8, "phone_reported": 8, "phone_monthly": 55,
        "insurance_on_time": 0, "insurance_reported": 0, "insurance_monthly": 0,
        "annual_income": 42000, "loan_amount": 6000, "loan_term_months": 24,
    },
    {
        "id": "NC-003", "months_in_country": 26, "bureau_status": "thin",
        "credit_score": 640,
        "rent_on_time": 24, "rent_reported": 26, "rent_monthly": 1500,
        "utilities_on_time": 22, "utilities_reported": 24, "utilities_monthly": 160,
        "phone_on_time": 20, "phone_reported": 24, "phone_monthly": 65,
        "insurance_on_time": 24, "insurance_reported": 26, "insurance_monthly": 130,
        "annual_income": 67000, "loan_amount": 12000, "loan_term_months": 36,
    },
    {
        "id": "NC-004", "months_in_country": 8, "bureau_status": "no_file",
        "credit_score": None,
        "rent_on_time": 5, "rent_reported": 8, "rent_monthly": 1500,
        "utilities_on_time": 3, "utilities_reported": 6, "utilities_monthly": 150,
        "phone_on_time": 4, "phone_reported": 6, "phone_monthly": 60,
        "insurance_on_time": 0, "insurance_reported": 0, "insurance_monthly": 0,
        "annual_income": 30000, "loan_amount": 8000, "loan_term_months": 36,
    },
    {
        "id": "NC-005", "months_in_country": 20, "bureau_status": "thin",
        "credit_score": 590,
        "rent_on_time": 12, "rent_reported": 20, "rent_monthly": 1300,
        "utilities_on_time": 9, "utilities_reported": 18, "utilities_monthly": 140,
        "phone_on_time": 10, "phone_reported": 20, "phone_monthly": 60,
        "insurance_on_time": 0, "insurance_reported": 0, "insurance_monthly": 0,
        "annual_income": 52000, "loan_amount": 9000, "loan_term_months": 36,
    },
]


def run() -> list[dict]:
    rows = []
    for app in SAMPLE_NEWCOMERS:
        signal = get_alt_payment_signal(app)
        outcome = adjudicate_newcomer(app, alt_signal=signal)
        rows.append({"id": app["id"], **outcome})
    return rows


def _print_table(rows: list[dict]) -> None:
    header = f"{'id':<8}{'bureau':<9}{'alt-score':>10}{'alt-DTI':>9}   {'decision':<16}"
    print(header)
    print("-" * len(header))
    for r, app in zip(rows, SAMPLE_NEWCOMERS):
        bureau = "no file" if app["bureau_status"] == "no_file" else f"thin {app['credit_score']}"
        score = "—" if r["alt_payment_score"] is None else f"{r['alt_payment_score']:.0f}"
        print(f"{r['id']:<8}{bureau:<9}{score:>10}{r['dti']:>8.0f}%   {r['decision']:<16}")
    print()
    for r in rows:
        print(f"  {r['id']}  →  {r['decision']}")
        print(f"      {r['reason']}")
        if r["decision"] == "REFER":
            print(f"      analyst sees: {r['analyst_summary']}")
        print()


if __name__ == "__main__":
    _print_table(run())
