"""
src/newcomer/alt_payment.py
────────────────────────────
STEP 1 — a REAL alternative-payment score for the newcomer segment.

WHY THIS EXISTS
───────────────
The platform already has an `alt_data_score` (src/data/alt_data.py), but it is
~40% anchored to the bureau credit score and built entirely from bureau fields.
It is therefore collinear with the bureau score, its information value is 0.015
(below the 0.02 selection floor), and it is dropped before the model ever runs.
In other words: it needs the very data a newcomer does not have.

This module computes an alternative-payment score from ONLY on-time payment
behaviour across four recurring obligations — rent, utilities, phone, insurance
— and works with NO bureau file. That independence is the whole point: it
carries signal exactly where the bureau is silent.

HOW THE SCORE IS BUILT (0–100)
──────────────────────────────
    per-stream on-time ratio  = payments_on_time / payments_reported
    weighted ratio            = Σ (weight_s · ratio_s) / Σ weight_s   over present streams
    maturity                  = min(months_observed / FULL_HISTORY_MONTHS, 1)
    score                     = 100 · weighted_ratio · (FLOOR + (1-FLOOR)·maturity)

All weights, the maturity floor, and the horizon live in config.py.

SWAP SEAM (synthetic today → real feed tomorrow)
────────────────────────────────────────────────
`get_alt_payment_signal()` is the single function to replace in production.
Today its body computes the score from the applicant's (synthetic) bill fields.
In a live deployment, swap the body for one call to a consumer-permissioned
feed — Nova Credit (cross-border), Equifax NeoBureau, or a rent reporter like
FrontLobby / Borrowell — returning the SAME dict shape. Nothing downstream
changes.

EXPECTED APPLICANT FIELDS
─────────────────────────
For each stream in {rent, utilities, phone, insurance}:
    <stream>_on_time    int   payments made on time
    <stream>_reported   int   payments that could have been made (0 = no such account)
A stream with `reported == 0` is treated as absent (the applicant has no such
tradeline), not as a stream of zero on-time payments.
"""

from __future__ import annotations
import logging

from src.newcomer import config

log = logging.getLogger(__name__)

STREAMS = ("rent", "utilities", "phone", "insurance")


def _stream_ratio(applicant: dict, stream: str) -> tuple[float, int]:
    """
    On-time ratio and months reported for one payment stream.

    Returns (ratio, reported). ratio is in [0, 1]; reported is the number of
    payment periods observed. reported == 0 means the stream is absent.
    """
    reported = int(applicant.get(f"{stream}_reported", 0) or 0)
    if reported <= 0:
        return 0.0, 0
    on_time = int(applicant.get(f"{stream}_on_time", 0) or 0)
    on_time = max(0, min(on_time, reported))          # clamp to [0, reported]
    return on_time / reported, reported


def compute_alt_payment_score(applicant: dict) -> dict:
    """
    Compute the alternative-payment score from bill-payment behaviour alone.

    Args:
        applicant: dict with the per-stream fields documented in the module
                   header. No bureau score is required or used.

    Returns:
        dict:
          alt_payment_score   float  0–100 (None if no streams at all)
          alt_tradelines      int    how many of the four streams are present
          months_observed     int    longest bill history (the observation window)
          per_stream          dict   {stream: {"on_time": n, "reported": m,
                                                "ratio": r}} for present streams
          source              str    "synthetic" (swap to the real provider name)
    """
    weights = config.ALT_SCORE_WEIGHTS

    present: dict[str, dict] = {}
    weighted_ratio_num = 0.0
    weight_sum = 0.0
    months_observed = 0

    for stream in STREAMS:
        ratio, reported = _stream_ratio(applicant, stream)
        if reported == 0:
            continue
        w = weights.get(stream, 0.0)
        present[stream] = {
            "on_time": int(applicant.get(f"{stream}_on_time", 0) or 0),
            "reported": reported,
            "ratio": round(ratio, 4),
        }
        weighted_ratio_num += w * ratio
        weight_sum += w
        months_observed = max(months_observed, reported)

    if weight_sum == 0.0:
        # No alternative tradelines at all — nothing to score on.
        return {
            "alt_payment_score": None,
            "alt_tradelines": 0,
            "months_observed": 0,
            "per_stream": {},
            "source": "synthetic",
        }

    weighted_ratio = weighted_ratio_num / weight_sum
    maturity = min(months_observed / config.FULL_HISTORY_MONTHS, 1.0)
    score = 100.0 * weighted_ratio * (
        config.MATURITY_FLOOR + (1.0 - config.MATURITY_FLOOR) * maturity
    )

    return {
        "alt_payment_score": round(score, 1),
        "alt_tradelines": len(present),
        "months_observed": months_observed,
        "per_stream": present,
        "source": "synthetic",
    }


def get_alt_payment_signal(applicant: dict) -> dict:
    """
    THE SWAP SEAM.

    Today: compute the signal from the applicant's synthetic bill-payment fields.

    Production: replace the body with a single call to a consumer-permissioned
    alternative-data feed, returning the SAME dict shape, e.g.:

        return nova_credit.fetch(applicant)          # cross-border bureau
        # or equifax_neobureau.fetch(applicant)
        # or frontlobby.fetch(applicant)             # rent reporting

    Keeping every caller on this one function means the synthetic→real switch is
    a one-line change with no downstream impact.
    """
    signal = compute_alt_payment_score(applicant)
    log.debug("alt-payment signal (%s): score=%s tradelines=%s",
              signal["source"], signal["alt_payment_score"],
              signal["alt_tradelines"])
    return signal
