"""
build.py
─────────
One-command build script for the credit risk platform.

Usage:
    python build.py              # Full pipeline
    python build.py --from 4    # Resume from Phase 4
    python build.py --phase 6   # Run single phase

Requires:
    data/raw/lending_club_loans.csv   (from Kaggle or synthetic fallback)
    data/raw/heloc_dataset_v1.csv     (from Kaggle or synthetic fallback)
"""

import subprocess
import sys
import time
from pathlib import Path

PHASES = [
    (1, "notebooks/phase1/01_phase1_data_pipeline.py",
     "Data Acquisition & Harmonization"),
    (3, "notebooks/phase3/03_feature_engineering.py",
     "Feature Engineering & WoE Binning"),
    (4, "notebooks/phase4/04_model_training.py",
     "Model Training (PD + LGD + EAD)"),
    (5, "notebooks/phase5/05_explainability.py",
     "Explainable AI (SHAP + Fairness)"),
    (6, "notebooks/phase6/06_compliance.py",
     "Regulatory & Compliance Analytics"),
    (9, "notebooks/phase9/09_fraud.py",
     "Fraud Detection & Monitoring"),
]


def run_phase(num: int, path: str, name: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  Phase {num}: {name}")
    print(f"{'='*60}")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, path],
        capture_output=False
    )

    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✓ Phase {num} complete ({elapsed:.0f}s)")
        return True
    else:
        print(f"  ✗ Phase {num} FAILED — check output above")
        return False


def main():
    # Parse args
    from_phase  = 1
    single      = None

    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--from" and i + 1 < len(sys.argv) - 1:
            from_phase = int(sys.argv[i + 2])
        elif arg == "--phase" and i + 1 < len(sys.argv) - 1:
            single = int(sys.argv[i + 2])

    print("\nCredit Risk Platform — Build Pipeline")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working directory: {Path.cwd()}")

    # Check data files
    lc_path    = Path("data/raw/lending_club_loans.csv")
    heloc_path = Path("data/raw/heloc_dataset_v1.csv")

    if lc_path.exists() and heloc_path.exists():
        print("\n  Real data files found — running on real data")
    else:
        print("\n  Real data files NOT found — using synthetic fallback")
        print("  For real data:")
        print("    LendingClub: kaggle.com/datasets/wordsforthewise/lending-club")
        print("    HELOC:       kaggle.com/datasets/averkiyoliabev/"
              "home-equity-line-of-creditheloc")

    t_start    = time.time()
    all_passed = True

    for num, path, name in PHASES:
        if single is not None and num != single:
            continue
        if num < from_phase:
            continue

        success = run_phase(num, path, name)
        if not success:
            all_passed = False
            print(f"\n  Build stopped at Phase {num}. Fix errors and re-run.")
            print(f"  Resume with: python build.py --from {num}")
            sys.exit(1)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    if all_passed:
        print(f"  BUILD COMPLETE ({total:.0f}s)")
        print(f"{'='*60}")
        print("""
  Launch the app:
    streamlit run src/app/streamlit_app.py

  Launch the API:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

  Enable Ollama (optional):
    ollama pull llama3 && ollama serve
        """)
    else:
        print(f"  BUILD FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
