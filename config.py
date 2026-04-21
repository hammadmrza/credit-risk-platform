"""
config.py
─────────
Central configuration for the Credit Risk Platform.
All paths, constants, and project-wide settings live here.
Import this module at the top of every notebook and script.
"""

from pathlib import Path

# ── Project root (works regardless of where script is run from) ──
ROOT = Path(__file__).parent

# ── Data directories ─────────────────────────────────────────────
DATA_DIR       = ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
EXTERNAL_DIR   = DATA_DIR / "external"

# ── Output directories ───────────────────────────────────────────
MODELS_DIR     = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"
ARTIFACTS_DIR  = ROOT / "artifacts"

for d in [RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR,
          MODELS_DIR, REPORTS_DIR, ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Raw data file paths ──────────────────────────────────────────
# User downloads these files and places them here.
# See data/README.md for download instructions.
LC_RAW_PATH    = RAW_DIR / "lending_club_loans.csv"
HELOC_RAW_PATH = RAW_DIR / "heloc_dataset_v1.csv"

# ── Processed data file paths ────────────────────────────────────
LC_CLEAN_PATH       = PROCESSED_DIR / "lending_club_clean.parquet"
HELOC_CLEAN_PATH    = PROCESSED_DIR / "heloc_clean.parquet"
HARMONIZED_PATH     = PROCESSED_DIR / "credit_model_dataset.parquet"
TRAIN_PATH          = PROCESSED_DIR / "train.parquet"
TEST_PATH           = PROCESSED_DIR / "test.parquet"

# ── Model artifact paths ─────────────────────────────────────────
SCORECARD_PATH      = MODELS_DIR / "scorecard_model.pkl"
XGB_PD_PATH         = MODELS_DIR / "xgb_pd_model.pkl"
LGD_MODEL_PATH      = MODELS_DIR / "lgd_model.pkl"
EAD_PARAMS_PATH     = MODELS_DIR / "ead_params.pkl"
BINNING_PATH        = MODELS_DIR / "binning_process.pkl"
SCALER_PATH         = MODELS_DIR / "scaler.pkl"

# ── LendingClub processing constants ─────────────────────────────
LC_TARGET_STATUSES  = ["Fully Paid", "Charged Off"]
LC_BAD_STATUSES     = ["Charged Off"]
LC_SAMPLE_SIZE      = 200_000       # Stratified sample for manageable size
LC_RANDOM_STATE     = 42

# ── HELOC processing constants ───────────────────────────────────
HELOC_BAD_VALUE     = "Bad"         # RiskPerformance == "Bad" → default = 1
HELOC_MISSING_CODES = [-7, -8, -9]  # HELOC encodes missing as negative integers

# ── Harmonized dataset constants ─────────────────────────────────
PRODUCT_TYPE_UNSECURED = 0          # LendingClub
PRODUCT_TYPE_SECURED   = 1          # HELOC

# ── Score calibration (PDO approach) ─────────────────────────────
PDO          = 20       # Points to double the odds
BASE_SCORE   = 600      # Score at base_odds
BASE_ODDS    = 50       # Good:Bad ratio at base_score
SCORE_MIN    = 300
SCORE_MAX    = 850

RISK_TIER_THRESHOLDS = {
    "A": (720, 850),    # Very low risk
    "B": (680, 719),    # Low risk
    "C": (630, 679),    # Moderate risk — grey zone
    "D": (580, 629),    # Elevated risk
    "E": (300, 579),    # High risk
}

# ── Train / test split ───────────────────────────────────────────
# Temporal split: train on older vintages, test on newer
# LendingClub: train 2007-2015, test 2016-2018
LC_TRAIN_CUTOFF_YEAR  = 2016
TEST_SIZE             = 0.20

# ── Alternative data composite ───────────────────────────────────
ALT_DATA_RANDOM_STATE = 42
ALT_DATA_NOISE_LEVEL  = 0.15    # Realistic signal-to-noise ratio

# ── PSI / CSI thresholds ─────────────────────────────────────────
PSI_GREEN   = 0.10   # No action required
PSI_AMBER   = 0.25   # Investigate
PSI_RED     = 0.25   # Mandatory model review (OSFI E-23 trigger)

# ── Basel III parameters ─────────────────────────────────────────
BASEL_RETAIL_CORRELATION_MIN = 0.03
BASEL_RETAIL_CORRELATION_MAX = 0.16
BASEL_CAPITAL_REQUIREMENT    = 0.08     # Pillar 1 minimum
Basel_CONFIDENCE_LEVEL       = 0.999    # 99.9% VaR

# ── IFRS 9 parameters ────────────────────────────────────────────
IFRS9_STAGE2_PD_THRESHOLD    = 0.20     # Significant credit deterioration
IFRS9_STAGE2_SCORE_DROP      = 30       # Points dropped since origination
IFRS9_STAGE3_DPD_THRESHOLD   = 90       # Days past due → credit-impaired
IFRS9_DISCOUNT_RATE          = 0.05     # EIR for ECL discounting

# ── MLflow ───────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "credit-risk-platform"
MLFLOW_TRACKING_URI    = str(ROOT / "mlruns")

# ── Ollama ───────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "llama3"
OLLAMA_TIMEOUT   = 60       # seconds

# ── FastAPI ──────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
