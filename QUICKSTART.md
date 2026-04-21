# Quickstart

From zero to a running Streamlit app in 30-45 minutes (most of which is the
pipeline build).

---

## Prerequisites

- **Python 3.12** (3.11 also works)
- **4 GB RAM minimum** (8 GB recommended for the full 2.2M-row LendingClub build)
- **~3 GB free disk** (raw data + processed parquet + model artifacts)
- A Kaggle account (to download LendingClub)
- Optional: [Ollama](https://ollama.ai) for LLM-drafted credit memos

---

## Step 1 — Clone and install

```bash
git clone https://github.com/hammadmrza/credit-risk-platform
cd credit-risk-platform
pip install -r requirements.txt
```

Installs: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optbinning`, `shap`,
`streamlit`, `plotly`, `fastapi`, `uvicorn`, `joblib`, `httpx`.

---

## Step 2 — Download the two datasets

### 2a. LendingClub (unsecured — 2.2M rows, ~1.5 GB)

1. Go to [kaggle.com/datasets/wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Click **Download** (requires a free Kaggle account)
3. Unzip the download. You'll see a file called `accepted_2007_to_2018Q4.csv.gz`
4. Decompress it and **rename** to `lending_club_loans.csv`
5. Place it at `data/raw/lending_club_loans.csv`

> **Important:** the filename must be exactly `lending_club_loans.csv`.
> `config.py` line 32 hardcodes this path. The loader handles schema drift
> across LendingClub vintages automatically (logs a warning for any expected
> column that isn't present).

### 2b. FICO HELOC (secured — 10K rows, ~660 KB)

1. Go to [community.fico.com/s/explainable-machine-learning-challenge](https://community.fico.com/s/explainable-machine-learning-challenge)
2. Download `heloc_dataset_v1.csv`
3. Place it at `data/raw/heloc_dataset_v1.csv`

**Alternative mirror if the FICO link is unavailable:**
[kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc](https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc)

---

## Step 3 — Verify the setup

```bash
python -c "import config; print('OK' if config.LC_RAW_PATH.exists() and config.HELOC_RAW_PATH.exists() else 'MISSING')"
```

Expected output: `OK`. If you see `MISSING`, double-check the file placement
and filenames.

---

## Step 4 — Build the pipeline

```bash
python build.py
```

Runs all 9 phases end-to-end. Expected wall time: **15–25 minutes** on a
modern laptop.

```
Phase 1  Data Acquisition & Harmonization        ~2 min
Phase 2  Exploratory Data Analysis (EDA)         skipped by default
Phase 3  Feature Engineering & WoE Binning       ~3 min
Phase 4  Model Training (PD + LGD + EAD)         ~5 min
Phase 5  Explainable AI (SHAP, counterfactuals)  ~3 min
Phase 6  Regulatory Analytics (Basel, IFRS 9)    ~2 min
Phase 9  Fraud Detection & Monitoring            ~3 min
```

Progress logs are printed every chunk. If the process is interrupted, resume
from a specific phase:

```bash
python build.py --from 4
```

---

## Step 5 — (Optional) Train the v1.1 segmented models

The v1.1 per-product PD models are built separately as a challenger to the
unified v1.0 model. Running this populates real OOT AUC numbers in Tab 3,
Section 3:

```bash
python notebooks/phase4b/04b_segmented_models.py
```

Expected wall time: **5-10 minutes**. Produces:

- `models/xgb_pd_unsecured.pkl`
- `models/xgb_pd_secured.pkl`
- `reports/phase4/model_comparison_segmented.csv`

---

## Step 6 — Launch the app

```bash
streamlit run src/app/streamlit_app.py
```

Your browser opens at `http://localhost:8501`. Seven tabs render.

---

## Step 7 — (Optional) Launch the API

For programmatic scoring:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI docs at `http://localhost:8000/docs`.

---

## Step 8 — (Optional) Enable Ollama LLM for credit memos

The app uses Llama 3 locally via Ollama to draft credit memos and adverse-action
letters. If Ollama is not installed, structured templates are used as a
fallback — the app still works, just without the LLM layer.

```bash
# Install from https://ollama.ai
ollama pull llama3
ollama serve
```

Re-run the app. Memos and letters now route through Llama 3. No PII is
transmitted externally (PIPEDA-compliant).

---

## Post-build verification checklist

Once `build.py` completes, open the app and confirm:

- [ ] **Tab 1** — loan term dropdown shows only `[36, 60]` for Unsecured and
      only `[36]` for HELOC
- [ ] **Tab 2** — "Load Test Portfolio" button populates 194K rows; four
      filters work
- [ ] **Tab 3 Section 1** — combined XGBoost OOT AUC is sensible (0.67-0.72
      is expected given the `loan_term_months` proxy; much lower means a
      calibration issue)
- [ ] **Tab 3 Section 3** — segmented comparison shows Unsecured and Secured
      separately (requires Phase 4b)
- [ ] **Tab 3 Section 5** — calibration gap between mean-PD and actual-DR is
      under 2% (Platt scaling should give this for free)
- [ ] **Tab 4** — CAR reads **11.50%** (the assumption), not 26.87% or any
      other number. If you see something else, the v1.1 fix didn't apply.
- [ ] **Tab 4** — IFRS 9 stage rows are colour-coded green/amber/red vs
      industry coverage bands
- [ ] **Tab 5** — profit curve renders with annotated profit-max threshold
      and dollar P&L comparison vs current platform threshold
- [ ] **Tab 7** — 7-column KPI row shows `CAR (assumed)` at 11.50%

If any of these fail, check `CHANGELOG_v1.1.md` for what should be in place,
then re-run `build.py --from N` for the failing phase.

---

## Resuming a stopped build

The build is phase-based and state-preserving. If you interrupt `build.py`
mid-pipeline, you can resume from any phase:

```bash
python build.py --from 4    # resume from Phase 4 onward
python build.py --only 6    # run only Phase 6 (for example, to re-stress)
```

Each phase writes its outputs to `reports/phaseN/` and `models/` so subsequent
phases pick up where they left off.

---

## Troubleshooting

**"LendingClub raw file not found"** — Filename must be exactly
`lending_club_loans.csv`, and it must be in `data/raw/`. If your Kaggle
download gave you `accepted_2007_to_2018Q4.csv.gz`, decompress and rename.

**"These expected columns not found in file"** — Harmless. LendingClub changed
their schema over the years. The loader automatically uses only columns that
are present.

**"Streamlit: Module not found"** — Your Python is not using the `requirements.txt`
environment. Either activate a virtual environment first or re-run
`pip install -r requirements.txt`.

**"Tab 4 shows CAR 26.87%"** — You're running v1.0 code. Upgrade to v1.1.
See `CHANGELOG_v1.1.md` for the fix.

**"Ollama connection refused"** — Ollama server is not running. Start with
`ollama serve` in a separate terminal. Or ignore — the app falls back to
templates automatically.

**"Profit curve is empty"** — Portfolio data hasn't been scored yet. Run
`python build.py --from 4` to regenerate.

**Out of memory during Phase 1** — Reduce `chunksize` in
`src/data/lendingclub.py:77` from 50000 to 20000. Trades build time for RAM.

---

## What's next

Once the app is running, read [PRODUCT_GUIDE.md](PRODUCT_GUIDE.md) for a
tab-by-tab walkthrough of what each section does, what the metrics mean, and
how to interpret the outputs in a credit-risk context.

For model governance details (intended use, training data, monitoring triggers,
v1.1 challenger architecture), see [MODEL_CARD.md](MODEL_CARD.md).

For the formal credit policy in the voice a real lender's risk team would use,
see [CREDIT_POLICY.md](CREDIT_POLICY.md).
