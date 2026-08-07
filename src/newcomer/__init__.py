"""
src/newcomer/
─────────────
Newcomer (thin-file) credit segment — a self-contained bolt-on.

This package extends the credit risk platform to serve newcomers to Canada
who have a thin or non-existent bureau file (established within the last
3 years, or no file at all) but who pay rent, utilities, phone, and
insurance on time.

ISOLATION CONTRACT
──────────────────
Everything for this segment lives in this folder. No module outside
`src/newcomer/` is modified. The existing platform (harmonize, PD models,
the main scoring pipeline) is untouched — delete this folder and the
platform runs exactly as before.

MODULES
───────
    config       — the dials / assumptions for the segment, in plain sight
    alt_payment  — Step 1: a REAL rent/utility/phone/insurance score that
                   works with no bureau file (+ the production swap seam)
    policy       — Step 2: the no-bureau-file adjudication rule (REFER-first)
    demo         — runs a handful of sample newcomers end-to-end

Steps 3 (segment monitoring) and 4 (data generator + PD sub-model) are
scaffolded separately once Steps 1-2 are signed off.
"""

from src.newcomer.alt_payment import (
    compute_alt_payment_score,
    get_alt_payment_signal,
)
from src.newcomer.policy import (
    adjudicate_newcomer,
    compute_newcomer_dti,
)

__all__ = [
    "compute_alt_payment_score",
    "get_alt_payment_signal",
    "adjudicate_newcomer",
    "compute_newcomer_dti",
]
