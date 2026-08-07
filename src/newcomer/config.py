"""
src/newcomer/config.py
──────────────────────
The DIALS for the Newcomer segment — every tunable assumption in one place,
in plain sight. Nothing about this segment is hidden inside the logic.

Each value is tagged:
    (assumption)  — a modelling choice; change it to suit your risk appetite
    (cited)       — grounded in an external figure, with the source noted

This file is deliberately kept separate from the platform's root config.py so
the newcomer segment stays fully isolated.
"""

# ── Segment definition ───────────────────────────────────────────────
# A "newcomer / thin-file" applicant is one whose bureau file was
# established within this many months — or who has no bureau file at all.
NEWCOMER_MAX_BUREAU_AGE_MONTHS = 36        # (assumption) "within the last 3 years"

# ── Synthetic population base rates ──────────────────────────────────
# Used ONLY by the data generator (Step 4). They shape the pretend
# population; they do not affect how a real applicant is scored.
PCT_NO_FILE           = 0.40   # (assumption) share of newcomers with NO bureau file
PCT_THIN_FILE         = 0.60   # (assumption) share with a thin (<3yr) file
MONTHS_IN_COUNTRY_MIN = 3      # (assumption)
MONTHS_IN_COUNTRY_MAX = 36     # (assumption)
TYPICAL_ON_TIME_RATE  = 0.90   # (assumption) baseline share of bills paid on time
TARGET_DEFAULT_RATE   = 0.12   # (assumption — tune to risk appetite) share that default
#
# (cited) Statistics Canada, Economic and Social Reports, Sept 2023,
# catalogue 36-28-0001: recent immigrants (<2 yrs) are credit-invisible at
# 14.8% vs 7.5% for Canadian-born families. NOTE: relayed via a search-engine
# snippet of the article, not verified against the source page directly
# (the environment used to build this was blocked from statcan.gc.ca).

# ── Alternative-payment score (Step 1) ───────────────────────────────
# Weights across the four recurring obligations. MUST sum to 1.0.
# Independent of the bureau score BY DESIGN — that independence is the
# entire reason the segment exists.
ALT_SCORE_WEIGHTS = {
    "rent":      0.40,   # largest, most predictive recurring obligation
    "utilities": 0.20,
    "phone":     0.20,
    "insurance": 0.20,
}

# A perfect record over 3 months should count for less than a perfect record
# over 2 years. The score is damped by how long we have observed the applicant.
FULL_HISTORY_MONTHS = 12     # (assumption) months of history considered "mature"
MATURITY_FLOOR      = 0.60   # (assumption) score multiplier at zero history
#   score = 100 * weighted_on_time_ratio * (MATURITY_FLOOR + (1-FLOOR)*maturity)
#   maturity = min(months_observed / FULL_HISTORY_MONTHS, 1.0)

# ── Eligibility thresholds (Step 2 policy) ───────────────────────────
ALT_SCORE_REFER_MIN = 60     # (assumption) min alt-score for a newcomer to be
                             #   REFERRED to an analyst instead of declined
MIN_ALT_TRADELINES  = 2      # (assumption) need at least 2 of the 4 bill streams
MIN_HISTORY_MONTHS  = 6      # (assumption) need at least 6 months of payment history
MAX_DTI             = 50.0   # (assumption) same affordability ceiling as the core
                             #   platform's HARD_POLICY["max_dti"]

# ── Affordability (newcomer DTI) ─────────────────────────────────────
# Newcomer DTI is a TDS-style ratio: it INCLUDES rent and the new loan.
#   DTI = (rent + utilities + phone + insurance + new-loan payment) / monthly income
# The new-loan monthly payment is amortised at this assumed rate, since the
# final priced rate is not known at adjudication time.
ASSUMED_APR = 0.12           # (assumption) rate used only to size the loan payment
