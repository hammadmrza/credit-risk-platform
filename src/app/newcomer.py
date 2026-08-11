"""
src/app/newcomer.py
────────────────────
"Newcomer to Canada" lending-program eligibility — a self-contained ADD-ON.

This module is deliberately decoupled from the scoring engine. It reads
attributes the desk already collects and reports whether an applicant meets the
*credit-bureau* portion of the Newcomer to Canada program. It NEVER changes the
model, the score, the PD, or any decision — it is an informational overlay only.

The real program has two kinds of rules, and only one of them is a bureau signal
this platform can evaluate:

  • Documentary eligibility (NOT modelled — a human verifies documents):
      – Landed in Canada < 5 years (PR card, IMM5292 / IMM5688, refugee, or a
        Canadian citizen with a foreign passport), and
      – A valid Canadian (province / territory) driver's licence.

  • Credit-bureau criteria (evaluated here), qualifying via either lane:
      A. No-hit / zero-risk-score thin bureau — qualifies regardless of tenure.
         (A true no-hit has no score, so the *core model* cannot rate them — that
          needs an alternative-data-only path, noted as a roadmap item.)
      B. Risk score ≥ 620 AND no trades rated R2 / I2 / O2 or worse (i.e. no
         derogatory / delinquent trades) AND bureau tenure ≤ 3 years.

Support mechanism — the alternative-data score (ADS). For these thin / young
files the model already leans on ADS (the ``ads_x_thin_file`` interaction), so a
newcomer with limited bureau depth is supported rather than auto-penalised. This
add-on surfaces that; it does not add or change any model logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

MIN_RISK_SCORE = 620      # program score floor for the score lane (lane B)
MAX_BUREAU_YEARS = 3      # bureau may now be up to 3 years old (per program sheet)
LANDED_MAX_YEARS = 5      # documentary rule — informational only


@dataclass
class NewcomerResult:
    """Outcome of the credit-bureau eligibility check (informational)."""
    qualifies: bool
    lane: str                                   # which bureau lane applied
    checks: List[Tuple[str, bool, str]] = field(default_factory=list)  # (label, passed, detail)
    ads_support: str = ""                       # how ALT data supports this segment
    doc_requirements: List[str] = field(default_factory=list)          # documentary rules (human-verified)
    scorable_by_core_model: bool = True         # False for a true no-hit
    notes: List[str] = field(default_factory=list)


def _doc_requirements() -> List[str]:
    return [
        f"Landed in Canada < {LANDED_MAX_YEARS} years — PR card, IMM5292 / "
        "IMM5688, refugee, or Canadian citizen with a foreign passport",
        "Valid Canadian (province / territory) driver's licence",
    ]


def _ads_support(alt_data_score: Optional[float]) -> str:
    # Honest framing: in production, ADS (with the ads_x_thin_file interaction and
    # a drop-in commercial alt-data API) is the support mechanism for thin/young
    # files. In the CURRENT synthetic data ADS is largely redundant with
    # credit_score (IV below the selection threshold) and is dropped in feature
    # selection — so this reflects design intent, not a live model driver yet.
    base = ("the alternative-data score (ADS) is the intended support mechanism "
            "for thin / young files — via the ads_x_thin_file interaction and a "
            "drop-in commercial alt-data API. In the current synthetic demo ADS "
            "is largely redundant with the bureau score, so this shows the design "
            "intent rather than a live model driver.")
    if alt_data_score is None:
        return "For this segment, " + base
    return (f"ADS = {float(alt_data_score):.0f}/100. For this segment, " + base)


def evaluate(*,
             credit_score: Optional[float],
             derog_marks: int,
             bureau_tenure_years: float,
             no_hit: bool = False,
             thin_file: bool = False,
             alt_data_score: Optional[float] = None,
             min_score: float = MIN_RISK_SCORE,
             max_bureau_years: float = MAX_BUREAU_YEARS) -> NewcomerResult:
    """Evaluate the credit-bureau portion of the Newcomer-to-Canada program.

    Pure and side-effect free — returns a :class:`NewcomerResult` and never
    touches the scoring model. Immigration / licence eligibility is returned as
    ``doc_requirements`` for a human to verify; it is not evaluated here.
    """
    cs = float(credit_score) if credit_score is not None else None
    derog = int(derog_marks or 0)
    tenure = float(bureau_tenure_years or 0)
    docs = _doc_requirements()
    ads = _ads_support(alt_data_score)

    # ── Lane A — no-hit / zero-score thin bureau (tenure limit waived) ──
    if no_hit:
        return NewcomerResult(
            qualifies=True,
            lane="No-hit / zero-score bureau",
            checks=[("No-hit or zero-score thin bureau", True,
                     "Qualifies via the thin-bureau lane — the 3-year tenure "
                     "limit is waived for this lane.")],
            ads_support=ads,
            doc_requirements=docs,
            scorable_by_core_model=False,
            notes=["A true no-hit has no bureau score, and the core model "
                   "requires a score to run — so it cannot rate this applicant "
                   "today. Serving no-hits needs an alternative-data-only "
                   "scoring path (roadmap item)."],
        )

    # ── Lane B — clean young file with a qualifying score ──────────────
    checks: List[Tuple[str, bool, str]] = [
        (f"Bureau tenure ≤ {max_bureau_years:g} yrs",
         tenure <= max_bureau_years, f"{tenure:g} yr on file"),
        ("No derogatory / delinquent trades (R2 / I2 / O2 or worse)",
         derog == 0, f"{derog} derogatory mark(s)"),
        (f"Risk score ≥ {min_score:g}",
         cs is not None and cs >= min_score,
         f"score {cs:.0f}" if cs is not None else "no score on file"),
    ]
    qualifies = all(passed for _, passed, _ in checks)
    return NewcomerResult(
        qualifies=qualifies,
        lane="Clean young-file (score lane)",
        checks=checks,
        ads_support=ads,
        doc_requirements=docs,
        scorable_by_core_model=True,
        notes=[],
    )
