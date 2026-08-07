"""
src/newcomer/pipeline.py
────────────────────────
End-to-end newcomer segment run, tying Steps 1–4 together:

    generate population (Step 4 data)
      → adjudicate each applicant (Steps 1 + 2)
        → monitor the segment (Step 3)
          → train the PD sub-model scaffold (Step 4 model)

Run:
    python -m src.newcomer.pipeline
    python -m src.newcomer.pipeline 5000     # custom population size
"""

from __future__ import annotations
import sys

from src.newcomer.data import get_newcomer_data
from src.newcomer.monitoring import adjudicate_batch, segment_report, print_report
from src.newcomer.model import train_newcomer_pd


def run_pipeline(n: int = 3000, random_state: int = 42, real_path: str | None = None) -> dict:
    # Step 4 (data): real file if supplied, else synthetic population
    records = get_newcomer_data(path=real_path, n=n, random_state=random_state)

    # Steps 1 + 2: score and adjudicate everyone
    evaluated = adjudicate_batch(records)

    # Step 3: segment monitoring & fairness
    report = segment_report(evaluated)

    # Step 4 (model): PD sub-model scaffold
    model_result = train_newcomer_pd(records, random_state=random_state)

    return {"records": records, "evaluated": evaluated,
            "report": report, "model": model_result}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    out = run_pipeline(n=n)

    print_report(out["report"])

    m = out["model"]
    print("\n── Newcomer PD sub-model (SCAFFOLD) ─────────────────────────")
    print(f"status : {m['status']}")
    print(f"metrics: {m['metrics']}")
    print("coefficients (higher → more default risk):")
    for feat, coef in m["coefficients"].items():
        print(f"    {feat:<20} {coef:>+.4f}")
    print("────────────────────────────────────────────────────────────")
