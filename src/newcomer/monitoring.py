"""
src/newcomer/monitoring.py
──────────────────────────
STEP 3 — newcomer segment monitoring & fairness.

Given a batch of newcomers that have been adjudicated (and, in the synthetic
world, carry a known default outcome), this module answers three questions:

  1. INCLUSION      — how many newcomers did we rescue from an automatic
                      reject, i.e. reach REFER instead of DECLINE? This is the
                      commercial point of the whole segment.

  2. RISK SEPARATION — does the alternative-payment score actually work? Do low
                      scorers default more than high scorers? Reported as a
                      default-rate ladder by score band plus an AUC/KS.

  3. FAIRNESS       — are "no-file" newcomers treated worse than "thin-file"
                      ones beyond genuine risk? Uses the EEOC four-fifths rule
                      (ratio < 0.80 flags a disparity) — the same rule the core
                      platform's fairness audit uses (CREDIT_POLICY.md §8.4).

All inputs are plain dicts, so this stays fully decoupled from the core book.
"""

from __future__ import annotations
import logging

from src.newcomer.alt_payment import get_alt_payment_signal
from src.newcomer.policy import adjudicate_newcomer

log = logging.getLogger(__name__)

SCORE_BANDS = [(0, 60), (60, 75), (75, 90), (90, 100.1)]


def adjudicate_batch(records: list[dict]) -> list[dict]:
    """
    Run every record through Step 1 + Step 2 and attach the result.
    Keeps `bureau_status` and `default_flag` (if present) for reporting.
    """
    out = []
    for r in records:
        signal = get_alt_payment_signal(r)
        decision = adjudicate_newcomer(r, alt_signal=signal)
        out.append({
            "id": r.get("id"),
            "bureau_status": r.get("bureau_status"),
            "decision": decision["decision"],
            "alt_payment_score": decision["alt_payment_score"],
            "dti": decision["dti"],
            "default_flag": r.get("default_flag"),
        })
    return out


def _auc_ks(scores: list[float], defaults: list[int]) -> tuple[float, float]:
    """
    AUC and KS of the alt-payment score as a risk ranker. A HIGHER score should
    mean LOWER default, so we test the score against (1 - default).
    Pure-python (no sklearn dependency) so monitoring runs anywhere.
    """
    pairs = [(s, d) for s, d in zip(scores, defaults) if s is not None and d is not None]
    if not pairs:
        return float("nan"), float("nan")
    goods = [s for s, d in pairs if d == 0]
    bads = [s for s, d in pairs if d == 1]
    if not goods or not bads:
        return float("nan"), float("nan")

    # AUC via Mann–Whitney: P(score_good > score_bad). Good payers should score higher.
    wins = ties = 0
    for gb in bads:
        for gg in goods:
            if gg > gb:
                wins += 1
            elif gg == gb:
                ties += 1
    auc = (wins + 0.5 * ties) / (len(goods) * len(bads))

    # KS: max gap between cumulative good and bad distributions across score.
    xs = sorted(set(scores))
    n_g, n_b = len(goods), len(bads)
    ks = 0.0
    for x in xs:
        cg = sum(1 for s in goods if s <= x) / n_g
        cb = sum(1 for s in bads if s <= x) / n_b
        ks = max(ks, abs(cg - cb))
    return round(auc, 4), round(ks, 4)


def segment_report(evaluated: list[dict]) -> dict:
    """
    Build the monitoring report from `adjudicate_batch` output.
    Returns a nested dict (also nice to print via `print_report`).
    """
    n = len(evaluated)
    decisions = [e["decision"] for e in evaluated]
    mix = {d: decisions.count(d) for d in sorted(set(decisions))}

    referred = mix.get("REFER", 0)
    consideration_rate = referred / n if n else 0.0

    # ── Risk separation (uses known synthetic outcomes across ALL records) ──
    scored = [e for e in evaluated
              if e["alt_payment_score"] is not None and e["default_flag"] is not None]
    bands = []
    for lo, hi in SCORE_BANDS:
        grp = [e for e in scored if lo <= e["alt_payment_score"] < hi]
        if grp:
            dr = sum(e["default_flag"] for e in grp) / len(grp)
            bands.append({"band": f"{lo:.0f}–{hi:.0f}", "n": len(grp),
                          "default_rate": round(dr, 4)})
    auc, ks = _auc_ks([e["alt_payment_score"] for e in scored],
                      [e["default_flag"] for e in scored])

    # ── Fairness: REFER-rate, no-file vs thin (four-fifths rule) ──
    def refer_rate(status: str) -> tuple[int, float]:
        grp = [e for e in evaluated if e["bureau_status"] == status]
        if not grp:
            return 0, float("nan")
        return len(grp), sum(e["decision"] == "REFER" for e in grp) / len(grp)

    n_nofile, rr_nofile = refer_rate("no_file")
    n_thin, rr_thin = refer_rate("thin")
    rates = [r for r in (rr_nofile, rr_thin) if r == r]     # drop NaN
    four_fifths = (min(rates) / max(rates)) if len(rates) == 2 and max(rates) > 0 else float("nan")

    return {
        "n": n,
        "decision_mix": mix,
        "consideration_rate": round(consideration_rate, 4),
        "risk_separation": {"bands": bands, "auc": auc, "ks": ks},
        "fairness": {
            "refer_rate_no_file": None if rr_nofile != rr_nofile else round(rr_nofile, 4),
            "refer_rate_thin": None if rr_thin != rr_thin else round(rr_thin, 4),
            "n_no_file": n_nofile, "n_thin": n_thin,
            "four_fifths_ratio": None if four_fifths != four_fifths else round(four_fifths, 4),
            "flag": (four_fifths == four_fifths and four_fifths < 0.80),
        },
    }


def print_report(report: dict) -> None:
    print("── Newcomer segment monitoring ─────────────────────────────")
    print(f"Applicants:            {report['n']:,}")
    print(f"Decision mix:          {report['decision_mix']}")
    print(f"Consideration rate:    {report['consideration_rate']:.1%}  "
          f"(share rescued to REFER instead of auto-decline)")
    rs = report["risk_separation"]
    print(f"\nRisk separation (does the score work?)  AUC={rs['auc']}  KS={rs['ks']}")
    for b in rs["bands"]:
        print(f"    alt-score {b['band']:>8}   n={b['n']:>5}   "
              f"default rate {b['default_rate']:.1%}")
    fr = report["fairness"]
    print(f"\nFairness (no-file vs thin, four-fifths rule):")
    print(f"    REFER rate  no-file={fr['refer_rate_no_file']}  "
          f"thin={fr['refer_rate_thin']}  ratio={fr['four_fifths_ratio']}")
    print(f"    disparity flag: {'⚠ YES' if fr['flag'] else 'no'}")
    print("────────────────────────────────────────────────────────────")
