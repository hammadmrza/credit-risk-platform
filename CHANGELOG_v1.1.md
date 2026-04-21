# Credit Risk Platform — v1.1 Release Notes

## Summary

31 fixes across the Streamlit app (21 fixes) and the Executive Dashboard (10 fixes),
plus a new v1.1 per-product PD modelling pipeline. All changes preserve backward
compatibility with your existing `.pkl` artifacts and processed parquet files.

---

## New files

| File | Purpose |
|------|---------|
| `src/models/pd_model_segmented.py`              | v1.1 training module — per-product XGBoost PD models |
| `notebooks/phase4b/04b_segmented_models.py`     | Runner that trains Unsecured + Secured models and writes comparison CSV |
| `reports/phase4/model_comparison_segmented.csv` | Illustrative per-product AUC/KS/Gini (replaced when Phase 4b is run) |
| `INSTALLATION_GUIDE.md`                         | Where each file goes + verification checklist |
| `CHANGELOG_v1.1.md`                             | This file |

---

## Updated files

| File | What changed |
|------|-------------|
| `src/app/streamlit_app.py`          | All 21 tab fixes (Tabs 1-6) |
| `src/app/utils.py`                  | 6 new analytical helpers + segmented report loader |
| `src/app/dashboard.py`              | 10 Executive Dashboard fixes |
| `MODEL_CARD.md`                     | Added §6 loan_term_months proxy disclosure + v1.1 governance section |
| `reports/phase3/iv_table.csv`       | Added "notes" column flagging loan_term_months proxy |
| `reports/phase3/selected_features.txt` | Added proxy footnote |

---

## Streamlit app — fixes by tab

### Tab 1 — Application Assessment (3)
- Loan term dropdown restricted per product (36/60 for unsecured; 36 only for HELOC with caption)
- Fraud strip collapsed into an expander for LOW-tier alerts; prominent for MEDIUM/HIGH/CONFIRMED
- Applicant name validation blocks letter generation when name is blank

### Tab 2 — Batch Scoring (3)
- Four interactive filters: Product, Risk Tier, Decision, Min Expected Loss
- Error summary banner with expandable per-row failure details
- Product × Tier concentration pivot table; unified schema across test-portfolio and upload paths

### Tab 3 — Model Performance (5)
- ROC curve with diagonal random baseline
- KS separation curve with KS-max score annotation
- Interactive confusion matrix at user-selectable PD threshold (+ precision, recall, F1, FPR, business interpretation)
- Lift chart + cumulative gains chart by score decile
- v1.0 Unified vs v1.1 Segmented model comparison (Section 3)

### Tab 4 — Compliance (3, 1 CRITICAL)
- **CRITICAL FIX**: Capital Adequacy Ratio denominator corrected from EAD to RWA. Previous
  formula produced a meaningless number; now uses correct `Available Capital / RWA` with
  11.5% CET1 target as the demonstration assumption.
- IFRS 9 stage rows colour-coded against industry coverage bands (green/amber/red)
- Expanded disclosure on HELOC LGD stress assumption (Freddie Mac historical reference)

### Tab 5 — Risk-Based Pricing (4)
- Rate schedule now uses portfolio-derived median PDs per tier (was hardcoded)
- Market rate comparison chart with 6 Canadian lender benchmarks (prime, credit union, fintech, subprime, B-lender HELOC)
- Profit curve sweeping approval thresholds from 5% to 70% PD — identifies the profit-max cutoff
- Cost of funds sensitivity analysis showing how many tiers hit the 35% cap as CoF rises

### Tab 6 — Fraud Monitoring (3)
- Time-series fraud trend by origination cohort with spike detection
- Product × fraud-type drill-down (pivot + stacked bar)
- Loss_attributed tooltip and column-definitions block for the investigation queue

---

## Dashboard — fixes (`src/app/dashboard.py`)

1. **CRITICAL**: CAR denominator fixed (was `min_capital / EAD`, now `available_capital / RWA`),
   consistent with Tab 4. Previously displayed a meaningless 26.87%; now correctly shows 11.50%.
2. Empty-portfolio guard at the top of `render_dashboard` — surfaces a helpful warning if
   `portfolio_regulatory.parquet` hasn't been built yet, instead of crashing.
3. Missing-columns guard — warns if the portfolio is missing any of the 9 required columns.
4. KPI row expanded from 6 to 7 metrics to include the corrected CAR.
5. PD distribution chart now uses the real `APPROVAL_THRESHOLD` (0.28) from utils instead of
   a hardcoded 35%. Both lines now show separately: "Approval threshold (28%)" and
   "Refer band ceiling (35%)".
6. EL concentration chart now uses two distinct colours (blue for loan share, red for EL share)
   instead of the same tier palette with opacity, which was hard to distinguish.
7. Concentration ratio now returns NaN for empty tiers instead of a 0.001-floor that produced
   misleading huge ratios.
8. Stress test guard simplified — previous `if` was always true because `required_capital`
   is always present.
9. v1.1 segmented-models banner added to Model Monitoring view pointing reader to Tab 3.
10. Shared constants imported from `utils.py` instead of hardcoded, so the dashboard cannot
    drift out of sync with the rest of the app.

---

## Suggested git commit messages

```
v1.1: Fix Basel III CAR denominator in Tab 4 and Executive Dashboard

Previous formula computed min_capital / EAD which is not CAR. Correct formula
is available_capital / RWA. Demonstration assumption: available capital =
11.5% CET1 × RWA. See MODEL_CARD §v1.1 Governance for context.

- src/app/streamlit_app.py: Tab 4 CAR section
- src/app/dashboard.py:    Portfolio Health KPI row
```

```
v1.1: Add per-product PD models (Unsecured + Secured) as challengers

Removes the loan_term_months product-proxy effect in the v1.0 unified model.
Expected OOT AUC: Unsecured ~0.72, Secured ~0.76, vs 0.68 unified.

v1.0 unified model remains champion for cross-product decisioning due to
HELOC thin-data risk (10K rows) and operational overhead of dual governance.

- src/models/pd_model_segmented.py:              new training module
- notebooks/phase4b/04b_segmented_models.py:      runner
- reports/phase4/model_comparison_segmented.csv:  comparison output
- src/app/streamlit_app.py:                       Tab 3 Section 3
- MODEL_CARD.md:                                  v1.1 governance section
```

```
v1.1: Disclose loan_term_months product proxy in feature report and MODEL_CARD

High IV reflects the default-rate gap between two product books (36/60-month
unsecured vs imputed-36 HELOC), not a genuine term→default relationship for
HELOC. Neutralized at scoring time; documented per OSFI E-23.

- MODEL_CARD.md:                         §6 added under Limitations
- reports/phase3/iv_table.csv:           notes column added
- reports/phase3/selected_features.txt:  proxy footnote added
```

```
v1.1: Enhance validation surface — ROC, KS, confusion matrix, lift/gains, profit curve

Tab 3 now includes ROC and KS plots, interactive confusion matrix with
business-interpretation text, lift and gains charts. Tab 5 adds profit curve
with portfolio-derived optimal threshold and Canadian market rate comparison.

- src/app/streamlit_app.py:  Tabs 3 + 5
- src/app/utils.py:          6 new helpers
```

```
v1.1: Tab 2 batch scoring — filters, error summary, unified schema

Adds interactive filters (Product, Risk Tier, Decision, Min EL), per-row error
summary with expandable details, and Product × Tier concentration pivot. Unifies
column schema across test-portfolio and upload paths.
```

```
v1.1: Tab 6 — time series trend, drill-downs, loss_attributed documentation

Adds origination-cohort fraud rate trend with spike detection, product × fraud-type
drill-down table and stacked bar, and full column definitions for the investigation
queue including loss_attributed recovery methodology.
```

---

## Running the app

```bash
streamlit run src/app/streamlit_app.py
```

No new dependencies beyond what v1.0 required.
