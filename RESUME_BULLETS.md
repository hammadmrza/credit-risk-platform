# Resume Bullets — Credit Risk & Fraud Detection Platform

**Project:** Credit Risk & Fraud Detection Platform — End-to-end credit risk decisioning system

**Tech Stack:** Python · XGBoost · scikit-learn · SHAP · OptBinning · Streamlit · FastAPI · Ollama · Plotly

**GitHub:** github.com/hammadmrza/credit-risk-platform

---

## How to use this document

Three variants below — each tuned to a different role target. Copy the block that best matches the job you're applying for, paste into your resume's Projects section, and tailor the specific metrics line to match the job description's keywords.

Each variant is **5 bullets long** and designed to fit in a standard resume Projects block (~250-300 words). The first bullet is always the hook; the last bullet is always the quantified outcome.

---

## Variant A — Senior Product Manager (Fintech, Risk & Decisioning)

**Use for:** Senior PM roles at fintech lenders, credit-decisioning platforms, risk-tech companies (Borrowell, Brim Financial, KOHO, Fairstone, Equifax, TransUnion product roles). Leads with system design and stakeholder language.

```
CREDIT RISK & FRAUD DECISIONING PLATFORM  ·  Personal project  ·  2025–2026
github.com/hammadmrza/credit-risk-platform

• Architected an end-to-end credit risk decisioning system (not a single model) integrating
  PD/LGD/EAD modelling, fraud detection, explainable AI, and Basel III / IFRS 9 regulatory
  analytics into a unified 7-tab Streamlit application — reproducing the architecture of a
  commercial consumer lender.

• Designed a hierarchical decision engine (fraud gate → hard policy → credit model → refer
  band → approve) with ECOA/FCAC-compliant adverse-action handling, including suppression
  of non-bureau signals and counterfactual recourse paths ("reduce DTI to 28% → APPROVE") —
  the kind of underwriter-facing UX that separates a decisioning platform from a model demo.

• Trained PD models on 2.2M real LendingClub loans + 10K FICO HELOC records; identified a
  product-proxy issue in the unified model (loan_term_months absorbing signal), built a v1.1
  per-product segmented challenger that lifted OOT AUC from 0.68 to ~0.72/0.76 by product,
  and documented the champion/challenger governance in an OSFI E-23 model card.

• Built a portfolio-level pricing tab that sweeps approval thresholds against a profit curve,
  surfaces the profit-maximising cutoff, and enforces Canada's 35% Criminal Code APR cap
  (s.347) — grounding the model's output in commercial reality with six Canadian lender
  benchmarks (prime bank / credit union / fintech / subprime / B-lender HELOC).

• Delivered a complete documentation suite: README, Quickstart, Product Guide, formal Credit
  Policy (in CRO-voice, 14 pages, OSFI-aligned), Model Card, and API Guide — shipping both
  the code and the governance artifacts a real lender's risk committee would require.
```

**Tailoring notes for Senior PM:**
- If the role emphasizes data/ML: move the segmented challenger bullet up to position 2
- If the role emphasizes compliance: move the Credit Policy bullet up
- If it's a pricing/revenue role: move the profit curve bullet up to position 2

---

## Variant B — Credit Risk Manager / Fraud Manager

**Use for:** Credit Risk Manager, Senior Credit Analyst, Fraud Manager, Model Risk Management, Validation Analyst roles. Leads with regulatory language and risk-manager vocabulary.

```
CREDIT RISK & FRAUD DECISIONING PLATFORM  ·  Independent portfolio project  ·  2025–2026
github.com/hammadmrza/credit-risk-platform

• Built and documented a commercial-grade consumer-credit decisioning platform covering
  PD/LGD/EAD estimation, Basel III IRB capital (full K = LGD·N[...] formula), IFRS 9
  three-stage ECL provisioning with origination-PD proxy, and three-scenario macroeconomic
  stress testing — with all methodology explicitly disclosed per OSFI E-23 model governance
  expectations.

• Trained XGBoost PD models on 2.2M real LendingClub + 10K FICO HELOC records using
  Information-Value feature selection, Weight-of-Evidence binning (OptBinning), and Platt
  calibration; achieved OOT calibration gap within ±2% and KS > 0.30, with the full
  validation workbench (ROC, KS, calibration deciles, Brier, lift/gains, confusion matrix
  at threshold) exposed in a validation tab.

• Diagnosed and disclosed a model-risk finding: loan_term_months acting as a product-type
  proxy in the unified dual-book model, inflating its IV and depressing reported AUC;
  engineered the v1.1 segmented challenger architecture that removes the proxy and lifts
  per-product OOT AUC to ~0.72 (unsecured) and ~0.76 (HELOC) — exactly the kind of finding
  OSFI E-23 validators look for.

• Authored a formal Credit Policy document covering risk appetite with quantitative limits,
  hierarchical decisioning with ECOA/Reg B and FCAC-compliant adverse-action handling,
  override authority matrix, Basel III/IFRS 9/stress testing methodology, and signature
  blocks — written in the voice a CRO and external examiner would expect.

• Delivered an integrated fraud module (XGBoost on 14 fraud-specific features) with FPD
  lift analysis, investigation queue, and production-swap interface for LexisNexis RiskView
  and Equifax Fraud Shield — structured as a post-funding monitoring view plus an upstream
  pre-approval fraud gate that DECLINE_FRAUDs above 65% probability.
```

**Tailoring notes for Credit Risk / Fraud Manager:**
- For a Fraud Manager role specifically: lead with bullet 5 (the fraud module), move bullet 3 (the model-risk finding) to position 2
- For Credit Risk Manager: keep order as is
- For Model Risk Management / Validation: move bullet 3 to position 1

---

## Variant C — Academic (MA Supervisor / Graduate School Applications)

**Use for:** MA directed-readings write-up, PhD applications, academic conference submissions, professor/advisor-facing contexts. Leads with research methodology and framework adherence.

```
CREDIT RISK & FRAUD DECISIONING PLATFORM  ·  MA Information Systems & Technology  ·  2025–2026
github.com/hammadmrza/credit-risk-platform

• Designed and implemented an integrated credit-risk decisioning system covering the full
  modelling lifecycle — data harmonization across two heterogeneous loan books (LendingClub
  unsecured, FICO HELOC secured), feature engineering via Information-Value selection and
  Weight-of-Evidence binning, XGBoost PD estimation with Platt scaling, SHAP-based
  explainability, and counterfactual recourse generation — with all design decisions
  grounded in OSFI E-23 and Basel III IRB frameworks.

• Applied rigorous model validation: out-of-time (2016-2018) test splits on 194K held-out
  loans, calibration analysis at the decile level with Brier score and calibration gap,
  lift and cumulative gains curves, PSI/CSI feature stability monitoring, and fairness
  auditing across five segments using demographic parity and equalized odds (EEOC four-fifths
  rule as the disparity threshold).

• Conducted a methodological investigation into product-proxy leakage in combined-dataset
  models: diagnosed loan_term_months absorbing product-type signal in the unified XGBoost
  (IV 0.19 despite no economic basis for term→HELOC default correlation), then built and
  evaluated a v1.1 segmented challenger architecture that trains per-product models with
  proxy features excluded — demonstrating the classical trade-off between single-model
  comparability and per-segment discriminatory power.

• Produced a formal OSFI E-23-style Model Card documenting intended use, training data
  provenance, metrics by segment, monitoring triggers (PSI > 0.25 retraining threshold),
  known limitations, and a champion/challenger governance framework — alongside a full
  Credit Policy document and Product Guide that together form a reproducible governance
  artifact suitable for academic reference or commercial deployment.

• Stack: Python · scikit-learn · XGBoost · OptBinning · SHAP · Streamlit · FastAPI · Ollama
  local LLM for PIPEDA-compliant credit memo generation · FAISS-free architecture;
  deployed as a 7-tab analyst workbench with integrated executive dashboard for CRO / Board
  Risk Committee-level reporting.
```

**Tailoring notes for academic:**
- For ITEC 6002 directed-readings final paper: move bullet 3 (the methodology investigation) to position 1 — it's the most defensible academic contribution
- For PhD applications: emphasize bullet 2 (validation rigour) and bullet 3 (methodological investigation)
- For professor engagement (e.g. Prof. Yang): lead with the segmented-challenger analysis as it mirrors the kind of finding NLP-based analysis papers report

---

## Three-line version (for Resume summary section or cover letter)

For when you have very limited space — resume summary line, cover letter, or short blurbs:

```
Built an end-to-end credit risk decisioning platform on 2.2M real LendingClub + 10K FICO
HELOC records — integrated PD/LGD/EAD modelling, fraud detection, SHAP explainability, and
Basel III/IFRS 9 regulatory analytics into a 7-tab Streamlit application. Diagnosed and
remediated a product-proxy issue in the unified model; documented the champion/challenger
framework in an OSFI E-23 Model Card. Full Credit Policy, Product Guide, and API docs
shipped alongside.
```

---

## Interview talking points — one-sentence answers

**"Walk me through the project."**
*"It's a credit risk decisioning system — not a single model. It takes a loan application,
runs it through a hierarchical fraud → policy → credit → refer → approve engine, and
produces all the regulatory artifacts a Canadian non-bank lender needs: the score, PD, EL,
Basel capital, IFRS 9 provision, adverse reasons, counterfactual recourse paths, and a
Llama-3-drafted credit memo — all in one coherent 7-tab Streamlit app."*

**"What's the hardest part?"**
*"Keeping the regulatory artifacts consistent with the model outputs as the model evolved.
I had a v1.0 unified PD model that reported AUC 0.68 because loan_term_months was acting as
a product-type proxy. I built the v1.1 per-product segmented challenger to prove the fix
lifted AUC to 0.72/0.76, but I kept the v1.0 unified as champion because cross-product
score comparability matters for portfolio capital allocation. Documenting that governance
decision in the Model Card — the 'why we didn't just promote' narrative — was harder than
either model itself."*

**"What would you change if you did it again?"**
*"Two things. First, I'd build the v1.1 segmented architecture from day one — the unified
model was a false economy. Second, I'd integrate a real alt-data API (Equifax NeoBureau or
Borrowell) earlier, because my synthetic alt_data_score is 40% anchored to credit score
and therefore adds no independent signal. The code has the swap interface already, but
running with synthetic alt data for so long meant I under-invested in the thin-file
scoring story."*

**"Why should a hiring manager care?"**
*"Most portfolio projects train a model. This one reproduces how a real lender actually
decides — integrated credit, fraud, policy, explainability, and compliance in one workflow.
The Credit Policy, Model Card, and champion/challenger governance show I understand that
getting a model into production is more about the artifacts around it than the model
itself."*

---

## Metrics quick reference

For when an interviewer asks for specific numbers:

| Metric                           | Value                                    |
|----------------------------------|------------------------------------------|
| Training data                    | 2.2M LendingClub + 10K FICO HELOC        |
| OOT test set                     | 194,564 loans (2016-2018 vintages)       |
| v1.0 unified OOT AUC             | ~0.68 (depressed by product proxy)       |
| v1.1 segmented OOT AUC           | Unsecured ~0.72 · Secured ~0.76          |
| KS (both models)                 | > 0.30 (industry deployment threshold)   |
| Calibration gap                  | Within ±2% (IFRS 9 acceptable)           |
| Features selected                | 10 (IV ≥ 0.02 threshold)                 |
| Basel III CAR (assumed)          | 11.5% (D-SIB CET1 target)                |
| Criminal Code APR cap            | 35% (s.347, effective Jan 2025)          |
| Fraud model AUC (synthetic)      | Inflated by synthetic labels; real est. 0.72-0.85 |
| Documentation pages              | 8 documents totalling ~70 pages          |
| Lines of Python                  | ~5,000 across 44 .py files               |
| Tabs in Streamlit app            | 7 (6 core + Executive Dashboard)         |

---

*Resume Bullets v1.0 · April 2026 · Use whichever variant matches the role. Keep a copy of this doc for reference when tailoring individual applications.*
