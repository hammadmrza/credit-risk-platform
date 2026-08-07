"""
src/newcomer/model.py
─────────────────────
STEP 4 (model) — newcomer PD sub-model SCAFFOLD.

Mirrors the platform's per-segment approach (src/models/pd_model_segmented.py)
but for the newcomer segment, predicting default from the alternative-payment
features rather than bureau features.

⚠ THIS IS A SCAFFOLD, NOT A VALIDATED MODEL.
   It trains on SYNTHETIC labels by default. Real newcomer repayment data does
   not exist publicly, so these metrics only prove the plumbing works — they
   are NOT evidence of real-world performance and must NOT be used to approve
   anyone. The moment a real outcome file is supplied (via data.get_newcomer_data
   with a path), the same code trains on it and the metrics become meaningful.
   Until then the segment stays REFER-first (human decides), by design.

Interpretable by choice: logistic regression, so every coefficient is legible
to a validator (OSFI E-23). Swap for XGBoost later if lift justifies the
opacity — the interface here stays the same.
"""

from __future__ import annotations
import logging

import numpy as np

from src.newcomer.alt_payment import compute_alt_payment_score
from src.newcomer.policy import compute_newcomer_dti

log = logging.getLogger(__name__)

# Deliberately NON-redundant: the four bill ratios are already summarised by
# alt_payment_score, so including both would create collinearity and produce
# nonsensical coefficient signs. We keep four conceptually distinct drivers —
# payment quality, breadth, tenure, and affordability — so every coefficient is
# legible to a validator.
FEATURES = [
    "alt_payment_score",   # payment quality (already blends the four bills)
    "alt_tradelines",      # breadth — how many bill types are on record
    "months_observed",     # tenure — length of the payment history
    "dti",                 # affordability (TDS-style, includes rent + new loan)
]


def build_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Turn applicant records into an (X, y) matrix over FEATURES."""
    X, y = [], []
    for r in records:
        sig = compute_alt_payment_score(r)
        score = 50.0 if sig["alt_payment_score"] is None else sig["alt_payment_score"]
        X.append([
            score,
            sig["alt_tradelines"],
            sig["months_observed"],
            min(compute_newcomer_dti(r), 100.0),
        ])
        y.append(int(r.get("default_flag", 0)))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def _ks(y_true: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ys = y_true[order]
    cum_bad = np.cumsum(ys) / max(ys.sum(), 1)
    cum_good = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    return float(np.abs(cum_good - cum_bad).max())


def train_newcomer_pd(records: list[dict], random_state: int = 42) -> dict:
    """
    Train the scaffold PD sub-model on the given records.

    Returns dict: {model, features, coefficients, metrics, status}.
    `status` always flags the scaffold nature and whether labels were synthetic.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X, y = build_features(records)
    if len(np.unique(y)) < 2:
        return {"model": None, "features": FEATURES, "coefficients": {},
                "metrics": {}, "status": "SKIPPED — only one outcome class present"}

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(Xtr, ytr)

    p_te = model.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p_te))
    metrics = {
        "auc": round(auc, 4),
        "ks": round(_ks(yte, p_te), 4),
        "gini": round(2 * auc - 1, 4),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "default_rate": round(float(y.mean()), 4),
    }
    coeffs = dict(zip(FEATURES, [round(float(c), 4) for c in model.coef_[0]]))

    log.info("Newcomer PD scaffold — AUC=%.3f KS=%.3f (n=%d)",
             metrics["auc"], metrics["ks"], len(y))

    return {
        "model": model,
        "features": FEATURES,
        "coefficients": coeffs,
        "metrics": metrics,
        "status": ("SCAFFOLD — trained on SYNTHETIC labels; not validated; "
                   "do not use for approval. Retrain on real outcomes to make "
                   "these metrics meaningful."),
    }


if __name__ == "__main__":
    from src.newcomer.data import generate_newcomers
    res = train_newcomer_pd(generate_newcomers(2000))
    print("status  :", res["status"])
    print("metrics :", res["metrics"])
    print("coeffs  :", res["coefficients"])
