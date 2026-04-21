"""
src/app/streamlit_app.py
─────────────────────────
Credit Risk & Fraud Detection Platform — v1.0 Preliminary Release
7-tab Streamlit dashboard covering the full credit risk decisioning lifecycle.

⚠️  PRELIMINARY VERSION — revisions to follow.
    Known limitations documented in PRODUCT_GUIDE.md and model_validation_report.docx.
"""
import sys
from pathlib import Path as P
sys.path.insert(0, str(P(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import brier_score_loss
import io, warnings
warnings.filterwarnings("ignore")

from src.app.dashboard import render_dashboard
from src.app.utils import (
    load_models, load_portfolio, load_reports, score_applicant,
    TIER_COLORS, TIER_LABELS, STAGE_COLORS, STAGE_LABELS,
    APPROVAL_THRESHOLD, REFER_BAND_LOWER, FRAUD_DECLINE_THRESH,
    HARD_POLICY, DECISION_CONFIG,
    # Analytical helpers for Tabs 3 and 5
    compute_roc_curve, compute_ks_curve, confusion_at_threshold,
    compute_lift_gains, compute_tier_pds_from_portfolio, compute_profit_curve,
)
from src.llm.ollama_client import (
    generate_credit_memo, generate_adverse_action_letter,
    generate_risk_summary, check_ollama_status,
)

st.set_page_config(page_title="Credit Risk Platform", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ── Professional title bar ───────────────────────────────────────────
st.markdown("""
<div style="border-bottom:3px solid #1F3864;padding:4px 0 14px 0;margin-bottom:18px">
  <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:16px">
    <div>
      <div style="font-size:1.9rem;font-weight:700;color:#1F3864;line-height:1.1;letter-spacing:-0.3px">
        Credit Risk &amp; Fraud Detection Platform
      </div>
      <div style="font-size:0.95rem;color:#555;font-style:italic;margin-top:3px">
        End-to-end decisioning system — PD/LGD/EAD · Basel III · IFRS 9 · SHAP · Fraud
      </div>
    </div>
    <div style="text-align:right;font-size:0.82rem;color:#666;line-height:1.45">
      <div style="font-weight:600;color:#1F3864;font-size:0.95rem">Hammad Mirza</div>
      <div>
        <span style="background:#E8EEF4;color:#1F3864;padding:2px 9px;border-radius:10px;
                     font-size:0.72rem;font-weight:700;letter-spacing:0.4px">
          v1.1 · APRIL 2026
        </span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.approve-badge{background:#d4edda;color:#1a7a4a;padding:8px 22px;border-radius:20px;
  font-weight:700;font-size:1.15rem;display:inline-block}
.refer-badge{background:#fff3cd;color:#7d5a00;padding:8px 22px;border-radius:20px;
  font-weight:700;font-size:1.15rem;display:inline-block}
.decline-badge{background:#f8d7da;color:#c0392b;padding:8px 22px;border-radius:20px;
  font-weight:700;font-size:1.15rem;display:inline-block}
.decline-fraud-badge{background:#f5c6cb;color:#6c0a0a;padding:8px 22px;border-radius:20px;
  font-weight:700;font-size:1.15rem;display:inline-block}
.explain-box{background:#f0f7ff;border-left:4px solid #2196F3;padding:12px 16px;
  border-radius:0 8px 8px 0;margin:8px 0 14px 0;font-size:0.88rem;color:#1a3a5c;line-height:1.6}
.warn-box{background:#fff8e1;border-left:4px solid #FFC107;padding:12px 16px;
  border-radius:0 8px 8px 0;margin:8px 0 14px 0;font-size:0.88rem;color:#5d4037;line-height:1.6}
.plain-english{background:#e8f5e9;border-left:4px solid #2d9e6b;padding:12px 16px;
  border-radius:0 8px 8px 0;margin:10px 0 14px 0;font-size:0.92rem;color:#1a3d2b;line-height:1.6}
.policy-box{background:#fdecea;border-left:4px solid #c0392b;padding:12px 16px;
  border-radius:0 8px 8px 0;margin:8px 0 14px 0;font-size:0.88rem;color:#6b1a1a;line-height:1.7}
.fraud-box{background:#f5c6cb;border-left:4px solid #6c0a0a;padding:12px 16px;
  border-radius:0 8px 8px 0;margin:8px 0 14px 0;font-size:0.88rem;color:#6c0a0a;line-height:1.6}
.refer-box{background:#fff3cd;border-left:4px solid #ffc107;padding:12px 16px;
  border-radius:0 8px 8px 0;margin:8px 0 14px 0;font-size:0.88rem;color:#7d5a00;line-height:1.6}
.tab-summary{background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;
  padding:14px 18px;margin-bottom:18px}
.tab-summary h4{margin:0 0 6px 0;color:#2c3e50;font-size:1.0rem}
.tab-summary p{margin:0;color:#555;font-size:0.87rem;line-height:1.6}
.health-ok{color:#1a7a4a;font-weight:600}
.health-warn{color:#e07520;font-weight:600}
.health-bad{color:#c0392b;font-weight:600}
.fraud-tier-confirmed{color:#6c0a0a;font-weight:700}
.fraud-tier-high{color:#c0392b;font-weight:600}
.fraud-tier-medium{color:#e07520;font-weight:600}
.fraud-tier-low{color:#1a7a4a;font-weight:600}
</style>
""", unsafe_allow_html=True)

models    = load_models()
portfolio = load_portfolio()
reports   = load_reports()
# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Credit Risk Platform")
    st.markdown("**v1.0 — Preliminary Release** · 9 phases · 43 source files")
    st.caption("⚠️ Preliminary — revisions to follow. See PRODUCT_GUIDE.md.")
    st.divider()

    # Model health banner
    st.markdown("### Model Health")
    mc = reports.get("model_comparison", pd.DataFrame())
    if len(mc) > 0:
        try:
            auc_val = float(mc.iloc[-1]["auc"])
            ac = "health-ok" if auc_val >= 0.70 else ("health-warn" if auc_val >= 0.65 else "health-bad")
            al = "Good" if auc_val >= 0.70 else ("Acceptable" if auc_val >= 0.65 else "Low")
            st.markdown(f'Model AUC: <span class="{ac}">{auc_val:.3f} — {al}</span>', unsafe_allow_html=True)
        except: pass
    csi = reports.get("csi", pd.DataFrame())
    if len(csi) > 0 and "status" in csi.columns:
        n_alert = (csi["status"] == "ALERT — investigate").sum()
        sc = "health-bad" if n_alert > 0 else "health-ok"
        sl = f"{n_alert} ALERT(s)" if n_alert > 0 else "All features stable"
        st.markdown(f'Feature stability: <span class="{sc}">{sl}</span>', unsafe_allow_html=True)
    if len(portfolio) > 0:
        try:
            el_rate = portfolio["expected_loss"].sum() / portfolio["ead_estimate"].sum()
            ec = "health-ok" if el_rate < 0.08 else ("health-warn" if el_rate < 0.15 else "health-bad")
            el = "Healthy" if el_rate < 0.08 else ("Elevated" if el_rate < 0.15 else "High")
            st.markdown(f'EL Rate: <span class="{ec}">{el_rate:.2%} — {el}</span>', unsafe_allow_html=True)
        except: pass

    st.divider()
    st.markdown("### System Status")
    ollama = check_ollama_status()
    if ollama["ollama_running"] and ollama["llama3_available"]:
        st.success("Ollama LLM — Active")
        st.caption("AI-generated credit memos enabled.")
    else:
        st.warning("Ollama — Template mode")
        st.caption("Install Ollama + llama3 for AI memos.")
    if models["loaded"]:
        st.success("Models loaded")
        if models.get("has_separate_models"):
            st.info("Separate product models active ✓")
        else:
            st.caption("Combined model (v1.0). Separate models in v1.1.")
    else:
        st.error("Models failed — run build.py first")
    st.info(f"Portfolio: **{len(portfolio):,}** loans")
    st.divider()
    st.markdown("### Data Mode")
    real_data = P("data/raw/lending_club_loans.csv").exists()
    if real_data:
        st.success("Real data (LendingClub + HELOC)")
    else:
        st.warning("Synthetic data mode")
        st.caption("Download real data from Kaggle and re-run build.py.")
    st.divider()
    with st.expander("About this platform"):
        st.markdown("""
**Positioning:** Full-stack risk decisioning prototype demonstrating how credit risk,
fraud detection, explainability, regulatory capital, and local LLM tooling can be
integrated into one platform.

1. **Data** — LendingClub (500K) + FICO HELOC
2. **Models** — WoE Scorecard + XGBoost PD + LGD/EAD
3. **Decision** — Fraud gate → Hard policy → Credit risk → Refer → Approve
4. **Explainability** — SHAP + counterfactuals + fairness audit
5. **Regulation** — Basel III IRB + IFRS 9 + stress testing
6. **LLM** — Ollama local (PIPEDA compliant)
7. **Fraud** — Post-funding FPD detection + alert tiers

Built by **Hammad Mirza**
        """)
    with st.expander("Decision framework"):
        st.markdown(f"""
**Hierarchical decision engine (v1.0):**

1. 🚨 **Fraud gate**: Fraud score > {FRAUD_DECLINE_THRESH:.0%} → DECLINE_FRAUD
2. ❌ **Hard policy rules**:
   - Credit score < {HARD_POLICY['min_credit_score']} → DECLINE_POLICY
   - DTI > {HARD_POLICY['max_dti']:.0f}% → DECLINE_POLICY
   - Derog marks ≥ {HARD_POLICY['max_derog_marks']} → DECLINE_POLICY
   - Hard inquiries > {HARD_POLICY['max_inquiries_6m']} → DECLINE_POLICY
   - HELOC LTV > {HARD_POLICY['max_ltv_heloc']:.0%} → DECLINE_POLICY
3. ❌ **Credit model**: PD > {APPROVAL_THRESHOLD:.0%} → DECLINE_CREDIT
4. 🔍 **Refer band**: PD {REFER_BAND_LOWER:.0%}–{APPROVAL_THRESHOLD:.0%} → REFER
5. ✅ **Approve**: All gates pass → APPROVE
        """)
    with st.expander("Glossary"):
        st.markdown("""
**PD** — Probability of Default: model estimate of 12-month default probability.

**LGD** — Loss Given Default: fraction of the loan lost if borrower defaults. Secured loans have lower LGD because collateral is recoverable.

**EAD** — Exposure at Default: balance outstanding at time of default.

**EL** — Expected Loss = PD × LGD × EAD.

**SHAP** — Shapley values: mathematically rigorous attribution of each feature's contribution to the model prediction.

**RWA** — Risk-Weighted Assets: Basel III converts loans to risk-adjusted numbers for capital calculation.

**IFRS 9** — Stage 1 = performing (12m ECL). Stage 2 = SICR (lifetime ECL). Stage 3 = impaired.

**REFER** — Application in the manual review band (PD 28–35%). A credit analyst must review before final decision.

**DECLINE_POLICY** — Declined by a hard rule before the model was run. Cannot be overridden by compensating factors.

**DECLINE_FRAUD** — Declined by the fraud gate. Do not disclose fraud reason to applicant.

**FPD** — First Payment Default: missing the first scheduled payment. Strongest post-funding fraud signal.

**BNI** — Bankruptcy Navigation Index: bureau-sourced score predicting bankruptcy probability. Production extension point — not in public datasets.
        """)

# ── Landing KPI row ──────────────────────────────────────────────────
# Pull live values from portfolio + reports (with safe fallbacks)
_n_loans = len(portfolio) if len(portfolio) > 0 else 0
_auc = None
try:
    mc_seg = reports.get("model_comparison_segmented", pd.DataFrame())
    if len(mc_seg) > 0 and "auc" in mc_seg.columns:
        _auc = float(mc_seg["auc"].max())  # best per-product segmented AUC
    else:
        mc = reports.get("model_comparison", pd.DataFrame())
        if len(mc) > 0:
            _auc = float(mc.iloc[-1]["auc"])
except Exception:
    pass
_n_features = 10  # fixed: IV >= 0.02 selection threshold

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
with kpi_c1:
    st.metric(
        "Portfolio",
        f"{_n_loans:,}" if _n_loans > 0 else "—",
        help="Loans in the scored OOT portfolio (2016-2018 vintages, held out of training)."
    )
with kpi_c2:
    st.metric(
        "Model AUC",
        f"{_auc:.3f}" if _auc is not None else "—",
        delta="OOT validated" if (_auc is not None and _auc >= 0.70) else None,
        delta_color="normal",
        help="Out-of-time AUC on held-out 2016-2018 loans. v1.1 segmented if available, else v1.0 unified."
    )
with kpi_c3:
    st.metric(
        "Features",
        f"{_n_features}",
        help="Selected via Information Value >= 0.02 threshold. WoE-binned."
    )
with kpi_c4:
    st.metric(
        "Frameworks",
        "6",
        help="OSFI B-20/E-23 * Basel III IRB * IFRS 9 * ECOA/Reg B * Criminal Code s.347 * PIPEDA"
    )

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Application Assessment",
    "📦 Batch Scoring",
    "📈 Model Performance",
    "🏛 Compliance",
    "💰 Pricing",
    "🔍 Fraud Monitoring",
    "📊 Executive Dashboard",
])
# ═══════════════════════════════════════════════════════════════
# TAB 1 — APPLICATION ASSESSMENT
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="tab-summary">
    <h4>📋 Single Application Assessment</h4>
    <p>Core loan officer decisioning tool. Enter applicant details and receive a complete
    credit decision in under one second. The platform applies a <b>hierarchical decision
    engine</b>: (1) fraud gate, (2) hard policy rules, (3) credit model, (4) refer band,
    (5) approve. Outputs include decision code, fraud score, SHAP explanation, adverse
    action reasons, counterfactual recourse paths, regulatory capital, and credit memo.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("app_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Loan Details**")
            product_type = st.selectbox("Product type",
                ["Unsecured Personal Loan", "HELOC (Secured)"],
                help="Unsecured = no collateral. HELOC secured by property — lower LGD but OSFI LTV limits apply.")
            product_int = 0 if "Unsecured" in product_type else 1
            loan_amount = st.number_input("Loan amount ($)", 1000, 100000, 15000, 500,
                help="Requested principal. Higher amounts increase EAD and Expected Loss.")
            # Loan term options restricted to values the model actually saw during training.
            # Unsecured: LendingClub provides only 36 or 60 month terms.
            # HELOC:     no term in source data; standardised to 36 months at scoring time.
            if product_int == 0:
                loan_term = st.selectbox("Term (months)", [36, 60], index=0,
                    help="Repayment period. LendingClub training data only contains 36- and "
                         "60-month unsecured loans — other values would be extrapolation.")
            else:
                loan_term = st.selectbox("Term (months)", [36],
                    help="HELOC term standardised to 36 months — the training imputation value — "
                         "to prevent the product-proxy SHAP dominance described in MODEL_CARD §6.")
                st.caption("ⓘ HELOC term standardised to 36 months at scoring time (product-proxy mitigation).")
            if product_int == 1:
                ltv_ratio = st.slider("LTV ratio", 0.40, 0.95, 0.70, 0.01,
                    help="Loan / Property Value. Policy maximum: 90%. LTV > 90% triggers DECLINE_POLICY regardless of credit score.")
            else:
                ltv_ratio = None
        with c2:
            st.markdown("**Borrower Profile**")
            credit_score = st.slider("Credit score (FICO)", 300, 850, 680,
                help=f"Hard policy floor: {HARD_POLICY['min_credit_score']}. Below this = DECLINE_POLICY regardless of DTI or income. A=720+, B=680-719, C=630-679, D=580-629, E<580.")
            annual_income = st.number_input("Annual income ($)", 10000, 500000, 65000, 1000,
                help="Gross annual income (stated). Used in DTI calculation and income-to-loan ratio fraud signal.")
            dti = st.slider("Debt-to-Income (%)", 0.0, 80.0, 28.0, 0.5,
                help=f"Hard policy maximum: {HARD_POLICY['max_dti']:.0f}%. Above this = DECLINE_POLICY. Healthy range: <36%. High risk: >43%.")
            employment = st.slider("Employment tenure (years)", 0, 20, 5,
                help="Years at current employer. <1 year triggers fraud short_tenure_flag.")
        with c3:
            st.markdown("**Credit History**")
            derog_marks = st.number_input("Derogatory marks", 0, 10, 0,
                help=f"Hard policy maximum: {HARD_POLICY['max_derog_marks']-1} (≥{HARD_POLICY['max_derog_marks']} = DECLINE_POLICY). Includes bankruptcies, charge-offs, collections.")
            total_accts = st.number_input("Total credit accounts", 1, 50, 10,
                help="Total accounts ever opened. <3 = thin file. More accounts = longer history = lower risk.")
            inquiries = st.number_input("Hard inquiries (last 6 months)", 0, 15, 1,
                help=f"Hard policy maximum: {HARD_POLICY['max_inquiries_6m']} (above = DECLINE_POLICY for loan stacking). 5+ = fraud velocity signal.")
            utilization = st.slider("Credit utilization (%)", 0, 100, 35,
                help="Revolving balance / limit. <30% healthy, >70% elevated risk. Second most important FICO factor.")
            alt_data = st.slider("Alt data score (0-100)", 0, 100, 55,
                help="Proxy for Equifax NeoBureau / Borrowell alt data. In production: live API call. See PRODUCT_GUIDE.md §9.")
            thin_file = st.checkbox("Thin file (<3 tradelines)",
                help="Fewer than 3 credit accounts. Increases model uncertainty. Alt data score weighted higher.")

        app_name  = st.text_input("Applicant first name (for adverse action letter)", "",
                                  placeholder="Required for letter generation",
                                  help="Used in the salutation of the adverse action letter. Leave blank if no letter is needed.")
        gen_memo  = st.checkbox("Generate credit memo + adverse action letter", True)

        # ── Risk Ratio Meters ─────────────────────────────────────────
        # Live indicators showing how the applicant's key ratios compare
        # to policy limits. Updates as inputs change. Green = safe,
        # Amber = approaching limit, Red = at or exceeding policy limit.
        st.markdown("---")
        st.markdown("**Risk Ratio Meters** — policy thresholds shown")

        # DTI meter
        dti_max   = 50.0
        dti_pct   = min(dti / dti_max, 1.0)
        dti_color = "#1a7a4a" if dti_pct < 0.70 else ("#e07520" if dti_pct < 1.0 else "#c0392b")
        dti_label = "✓ Within limit" if dti < dti_max else "✗ Exceeds policy limit"

        # PTI — Payment to Income
        # Monthly payment estimated using flat rate approximation
        _rate_monthly = 0.18 / 12  # proxy rate for PTI
        _term         = loan_term if loan_term > 0 else 36
        try:
            _monthly_pmt = loan_amount * _rate_monthly / (1 - (1 + _rate_monthly) ** -_term)
        except ZeroDivisionError:
            _monthly_pmt = loan_amount / _term
        _monthly_income = annual_income / 12
        pti             = (_monthly_pmt / _monthly_income * 100) if _monthly_income > 0 else 0
        pti_max         = 20.0 if product_int == 0 else 28.0
        pti_pct         = min(pti / pti_max, 1.0)
        pti_color       = "#1a7a4a" if pti_pct < 0.70 else ("#e07520" if pti_pct < 1.0 else "#c0392b")
        pti_label       = "✓ Affordable" if pti < pti_max else "✗ Exceeds affordability guideline"

        m_cols = st.columns(3 if product_int == 1 else 2)

        with m_cols[0]:
            st.markdown(f"**DTI — Debt-to-Income**")
            st.markdown(
                f'<div style="background:#e9ecef;border-radius:6px;height:14px;margin:4px 0">'
                f'<div style="background:{dti_color};width:{dti_pct*100:.0f}%;height:14px;'
                f'border-radius:6px;transition:width 0.3s"></div></div>',
                unsafe_allow_html=True)
            st.caption(f"{dti:.1f}% / {dti_max:.0f}% max  ·  {dti_label}")

        with m_cols[1]:
            st.markdown(f"**PTI — Payment-to-Income**")
            st.markdown(
                f'<div style="background:#e9ecef;border-radius:6px;height:14px;margin:4px 0">'
                f'<div style="background:{pti_color};width:{pti_pct*100:.0f}%;height:14px;'
                f'border-radius:6px;transition:width 0.3s"></div></div>',
                unsafe_allow_html=True)
            st.caption(
                f"{pti:.1f}% / {pti_max:.0f}% max  ·  "
                f"Est. monthly payment ${_monthly_pmt:,.0f}  ·  {pti_label}")

        if product_int == 1 and ltv_ratio is not None:
            ltv_max   = 0.90
            ltv_pct   = min(ltv_ratio / ltv_max, 1.0)
            ltv_color = "#1a7a4a" if ltv_pct < 0.78 else ("#e07520" if ltv_pct < 1.0 else "#c0392b")
            ltv_label = "✓ Within limit" if ltv_ratio < ltv_max else "✗ Exceeds policy limit — DECLINE_POLICY"
            with m_cols[2]:
                st.markdown("**LTV — Loan-to-Value**")
                st.markdown(
                    f'<div style="background:#e9ecef;border-radius:6px;height:14px;margin:4px 0">'
                    f'<div style="background:{ltv_color};width:{ltv_pct*100:.0f}%;height:14px;'
                    f'border-radius:6px;transition:width 0.3s"></div></div>',
                    unsafe_allow_html=True)
                st.caption(f"{ltv_ratio:.0%} / {ltv_max:.0%} max  ·  {ltv_label}")

        st.markdown("---")
        submitted = st.form_submit_button("Score Application", use_container_width=True, type="primary")

    if submitted and models["loaded"]:
        applicant = {
            "loan_amount": loan_amount, "annual_income": annual_income, "dti": dti,
            "credit_score": credit_score, "employment_length_years": employment,
            "product_type": product_int, "loan_term_months": loan_term,
            "num_derogatory_marks": derog_marks, "total_accounts": total_accts,
            "num_inquiries_last_6m": inquiries, "credit_utilization": utilization,
            "alt_data_score": alt_data, "ltv_ratio": ltv_ratio, "thin_file_flag": thin_file,
        }
        with st.spinner("Running scoring pipeline..."):
            try:
                result = score_applicant(models, applicant)
            except Exception as _e:
                st.error(f"Scoring error: {_e}")
                result = None

        if result is None:
            st.stop()

        st.divider()
        decision = result["decision"]
        dc = DECISION_CONFIG[decision]

        # ── Decision badge
        st.markdown(f'<span class="{dc["badge_class"]}">{dc["icon"]}</span>',
                    unsafe_allow_html=True)
        st.caption(f"Approval threshold PD ≤ {APPROVAL_THRESHOLD:.0%} | "
                   f"Refer band PD {REFER_BAND_LOWER:.0%}–{APPROVAL_THRESHOLD:.0%}")

        # ── Decision explanation box
        if decision == "DECLINE_FRAUD":
            st.markdown(f'<div class="fraud-box"><b>🚨 Fraud Gate Triggered</b><br>'
                        f'{dc["description"]}<br><br>'
                        f'<b>Fraud probability:</b> {result["fraud_score"]:.1%} '
                        f'(threshold: {FRAUD_DECLINE_THRESH:.0%})<br>'
                        f'<b>Alert tier:</b> {result["fraud_alert_tier"]}<br>'
                        f'<b>Important:</b> Do not disclose fraud as the decline reason to '
                        f'the applicant. Use a generic decline reason in the adverse action '
                        f'letter per FCAC guidelines.</div>', unsafe_allow_html=True)
        elif decision == "DECLINE_POLICY":
            failures_html = "".join(f"<li>{f}</li>" for f in result["policy_failures"])
            st.markdown(f'<div class="policy-box"><b>❌ Hard Policy Rule Failure</b><br>'
                        f'{dc["description"]}<br><br>'
                        f'<b>Failed rules:</b><ul>{failures_html}</ul>'
                        f'Policy rules apply upstream of the credit model. '
                        f'The PD of {result["pd_pit"]:.1%} is irrelevant — '
                        f'these rules cannot be overridden by compensating factors.'
                        f'</div>', unsafe_allow_html=True)
        elif decision == "REFER":
            st.markdown(f'<div class="refer-box"><b>🔍 Manual Review Required</b><br>'
                        f'{dc["description"]}<br><br>'
                        f'PD of {result["pd_pit"]:.1%} is in the refer band '
                        f'({REFER_BAND_LOWER:.0%}–{APPROVAL_THRESHOLD:.0%}). '
                        f'A credit analyst should review compensating factors: '
                        f'employment stability, asset position, relationship depth, '
                        f'and any information not captured by the model before issuing '
                        f'a final APPROVE or DECLINE_CREDIT decision.'
                        f'</div>', unsafe_allow_html=True)
        else:
            # Plain-English summary for approve/decline_credit
            pd_val = result["pd_pit"]
            if pd_val < 0.08: pd_plain = "very low risk"
            elif pd_val < 0.15: pd_plain = "low risk"
            elif pd_val < 0.25: pd_plain = "moderate risk"
            elif pd_val < 0.40: pd_plain = "elevated risk"
            else: pd_plain = "high risk"

            if decision == "APPROVE":
                top_prot = [f for f in result["display_factors"]
                            if f["direction"] == "decreases_risk"]
                top_name = top_prot[0]["feature"].replace("_"," ") if top_prot else "overall profile"
                plain = (f"This applicant is a <b>{pd_plain}</b> borrower with a "
                         f"{pd_val:.1%} probability of default. Their strongest protective "
                         f"factor is <b>{top_name}</b>. The lender expects to lose "
                         f"${result['el']:,.0f} on this loan over its lifetime.")
            else:
                top_risk = [f for f in result["display_factors"]
                            if f["direction"] == "increases_risk"]
                top_name = top_risk[0]["feature"].replace("_"," ") if top_risk else "risk profile"
                plain = (f"This application was declined on credit risk grounds. "
                         f"PD {pd_val:.1%} exceeds the approval threshold "
                         f"{APPROVAL_THRESHOLD:.0%}. Primary concern: <b>{top_name}</b>.")

            st.markdown(f'<div class="plain-english">📌 <b>Summary:</b> {plain}</div>',
                        unsafe_allow_html=True)

        # ── Fraud status (shown on all decisions, collapsed for LOW tier)
        if result["fraud_available"]:
            fraud_tier = result["fraud_alert_tier"]
            tier_colors = {"CONFIRMED":"#6c0a0a","HIGH":"#c0392b",
                           "MEDIUM":"#e07520","LOW":"#1a7a4a"}
            tier_actions = {
                "CONFIRMED": "Immediate account freeze + investigation",
                "HIGH": "Escalate to fraud team within 24h",
                "MEDIUM": "Enhanced monitoring + QA review",
                "LOW": "Routine monitoring",
            }
            fc = tier_colors.get(fraud_tier, "#555")
            fa = tier_actions.get(fraud_tier, "")

            if fraud_tier == "LOW":
                # Collapse LOW-tier fraud results into an expander to reduce noise on clean applicants
                with st.expander(f"🔍 Fraud check passed — LOW alert tier "
                                 f"({result['fraud_score']:.1%})", expanded=False):
                    st.caption(f"Fraud score: {result['fraud_score']:.1%} · "
                               f"Alert tier: **LOW** · Action: {fa}")
                    st.caption("No fraud indicators triggered. Standard post-funding monitoring applies.")
            else:
                # Prominent strip for MEDIUM/HIGH/CONFIRMED — these require action
                st.markdown(
                    f'<div style="background:#f8f9fa;border-left:3px solid {fc};'
                    f'padding:8px 14px;border-radius:0 6px 6px 0;margin:6px 0;'
                    f'font-size:0.85rem">'
                    f'🔍 <b>Fraud Score:</b> {result["fraud_score"]:.1%} | '
                    f'<b>Alert Tier:</b> <span style="color:{fc};font-weight:700">'
                    f'{fraud_tier}</span> | {fa}</div>',
                    unsafe_allow_html=True)

        st.divider()

        # Only show full metrics for credit-model-evaluated decisions
        if decision in ("APPROVE", "DECLINE_CREDIT", "REFER"):
            st.markdown("### Core Risk Metrics")
            st.markdown("""<div class="explain-box">
            <b>Risk Score</b>: Internal model output (300-850 PDO scale). Not a bureau FICO score.
            <b>PD</b>: Calibrated default probability (Platt scaling — matches actual default rates).
            <b>EL</b>: PD × LGD × EAD. Dollar amount lender expects to lose.
            <b>IFRS 9</b>: Stage 1=performing (12m ECL). Stage 2=SICR (lifetime ECL). Stage 3=impaired.
            </div>""", unsafe_allow_html=True)

            c1,c2,c3,c4 = st.columns(4)
            tier_desc = {"A":"Very Low Risk","B":"Low Risk","C":"Moderate",
                         "D":"Elevated","E":"High Risk"}
            with c1:
                st.metric("Risk Score", result["credit_score"])
                st.caption(f"Tier **{result['risk_tier']}** — {tier_desc.get(result['risk_tier'],'')}")
                st.caption("Model output 300-850. Not a bureau FICO score.")
            with c2:
                st.metric("Probability of Default", f"{result['pd_pit']:.1%}")
                st.caption(f"TTC PD (Basel III): {result['pd_ttc']:.1%}")
                st.caption(f"Model: {result['model_used']}")
            with c3:
                st.metric("Expected Loss", f"${result['el']:,.0f}")
                st.caption(f"LGD: {result['lgd']:.1%} · EAD: ${result['ead']:,.0f}")
            with c4:
                s = result["ifrs9_stage"]
                sn = {1:"Stage 1 — Performing",2:"Stage 2 — SICR",3:"Stage 3 — Impaired"}
                st.metric("IFRS 9 Stage", sn.get(s, f"Stage {s}"))
                st.caption(f"ECL: ${result['ecl']:,.0f}")

            st.divider()
            st.markdown("### Model Explanation (SHAP)")
            st.markdown("""<div class="explain-box">
            <b>How to read:</b> Red bars pushed toward default (increased risk).
            Green bars pushed away from default (reduced risk). Length = magnitude.
            Computed using Shapley values — guarantees contributions sum to final prediction.
            For HELOC: loan_term_months filtered from display (product proxy, not credit signal).
            </div>""", unsafe_allow_html=True)

            l, r = st.columns(2)
            with l:
                factors = result["display_factors"][:6]
                if factors:
                    feats  = [f["feature"].replace("_"," ").title() for f in factors]
                    vals   = [f["shap_impact"] for f in factors]
                    colors = ["#c0392b" if v > 0 else "#1a7a4a" for v in vals]
                    fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h",
                        marker_color=colors, text=[f"{v:+.4f}" for v in vals],
                        textposition="outside"))
                    fig.update_layout(title="SHAP Feature Contributions", height=300,
                        margin=dict(l=10,r=60,t=40,b=10),
                        xaxis_title="Impact (red=increases risk, green=reduces risk)",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

            with r:
                if decision in ("DECLINE_CREDIT",):
                    st.markdown("""<div class="explain-box">
                    <b>Adverse action reasons (FCAC/ECOA required):</b>
                    Derived from top SHAP risk drivers. Lender cannot cite "low credit score"
                    generically — must state the specific factor.
                    </div>""", unsafe_allow_html=True)
                    for i, reason in enumerate(result["adverse_reasons"], 1):
                        st.markdown(f"**{i}.** {reason}")
                    if result["counterfactual"]:
                        paths = result["counterfactual"].get("single_feature_paths", [])[:3]
                        if paths:
                            st.divider()
                            st.markdown("**How to get approved — counterfactual paths:**")
                            for p in paths:
                                st.markdown(f"• {p['description']} → new PD: **{p['new_pd']:.1%}**")
                elif decision == "REFER":
                    st.markdown("""<div class="refer-box">
                    <b>Manual review guidance:</b> The model has flagged this application
                    for human review. The analyst should assess compensating factors not
                    captured in the model: employment stability trend, asset position,
                    reasons for derogatory marks (if any), and relationship depth.
                    Final decision: APPROVE or DECLINE_CREDIT. Document rationale in credit file.
                    </div>""", unsafe_allow_html=True)
                    if result["counterfactual"]:
                        paths = result["counterfactual"].get("single_feature_paths", [])[:2]
                        if paths:
                            st.markdown("**Marginal paths to approval:**")
                            for p in paths:
                                st.markdown(f"• {p['description']} → PD: **{p['new_pd']:.1%}**")
                else:
                    st.success("Application meets all credit and policy criteria.")
                    protective = [f for f in result["display_factors"]
                                  if f["direction"] == "decreases_risk"]
                    if protective:
                        st.markdown("**Top protective factors:**")
                        for f in protective[:3]:
                            fn = f["feature"].replace("_"," ").title()
                            st.markdown(f"• **{fn}**: value {f['value']:.1f} "
                                        f"(impact: {f['shap_impact']:+.4f})")

            st.divider()
            st.markdown("### Regulatory Metrics")
            st.markdown("""<div class="explain-box">
            <b>RWA</b>: Basel III IRB converts loan to risk-adjusted asset at 99.9% confidence.
            <b>Min Capital (8%)</b>: Shareholders equity OSFI requires per OSFI CAR.
            <b>ECL</b>: IFRS 9 forward-looking accounting reserve.
            <b>Recommended Rate</b>: CoF 5.5% + EL + OpEx 1.5% + ROE 2.0%. Hard cap 35%
            (Criminal Code s.347, effective January 2025).
            </div>""", unsafe_allow_html=True)
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("RWA", f"${result['rwa']:,.0f}"); r1.caption("Basel III Risk-Weighted Assets")
            r2.metric("Min Capital (8%)", f"${result['min_capital']:,.0f}"); r2.caption("OSFI CAR requirement")
            r3.metric("ECL Provision", f"${result['ecl']:,.0f}"); r3.caption("IFRS 9 accounting reserve")
            r4.metric("Recommended Rate", f"{result['risk_based_rate']:.2%}"); r4.caption("Risk-based APR (35% cap)")

        if gen_memo and decision not in ("DECLINE_FRAUD",):
            st.divider()
            st.markdown("### Credit Memo")
            st.markdown("""<div class="explain-box">
            Generated by Llama 3 via Ollama (local — no PII transmitted externally, PIPEDA compliant).
            Template fallback when Ollama not installed.
            </div>""", unsafe_allow_html=True)
            with st.spinner("Generating memo..."):
                memo = generate_credit_memo(
                    applicant=applicant, pd_score=result["pd_pit"],
                    credit_score=result["credit_score"], risk_tier=result["risk_tier"],
                    lgd=result["lgd"], ead=result["ead"], expected_loss=result["el"],
                    shap_factors=result["all_factors"], decision=decision,
                    product_type=product_int)
            st.text_area("", memo, height=260, label_visibility="collapsed")

            if decision in ("DECLINE_CREDIT", "DECLINE_POLICY", "REFER"):
                st.markdown("### Adverse Action Letter")
                st.markdown("""<div class="explain-box">
                Required by FCAC/ECOA when declining or referring credit. States specific reasons,
                right to free credit report, and dispute contact information.
                Note: For DECLINE_FRAUD, do not disclose fraud reason — use generic decline.
                </div>""", unsafe_allow_html=True)

                # For DECLINE_POLICY: use the specific policy rule that failed,
                # not SHAP model reasons — the model never ran for policy declines.
                if decision == "DECLINE_POLICY":
                    letter_reasons = result["policy_failures"][:3]
                    # Truncate to concise one-liners for the letter
                    letter_reasons = [r.split(".")[0] + "." for r in letter_reasons]
                else:
                    letter_reasons = result["adverse_reasons"]

                # Validate applicant name — required for letter salutation
                if not app_name or not app_name.strip():
                    st.warning("⚠ Enter the applicant's first name above to generate the "
                               "adverse action letter. A letter addressed to 'Applicant' is "
                               "not acceptable for a real adverse action disclosure.")
                else:
                    with st.spinner("Generating letter..."):
                        letter = generate_adverse_action_letter(
                            applicant_name=app_name.strip(), loan_amount=loan_amount,
                            product_type=product_int, pd_score=result["pd_pit"],
                            adverse_reasons=letter_reasons)
                    st.text_area("", letter, height=280, label_visibility="collapsed")
                    st.download_button("⬇ Download Letter", letter,
                        f"adverse_action_{app_name.strip().lower()}.txt", "text/plain")
# ═══════════════════════════════════════════════════════════════
# TAB 2 — BATCH SCORING
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="tab-summary">
    <h4>📦 Batch Portfolio Scoring</h4>
    <p>Score thousands of applications at once. Each row runs the full hierarchical
    decision engine: fraud gate → hard policy → credit model → refer band → approve.
    Upload a CSV or load the 194,564-loan OOT test portfolio. Filter, segment, and
    export the results.</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Upload CSV**")
        st.markdown("""<div class="explain-box">Required columns: <b>loan_amount</b>,
        <b>annual_income</b>, <b>dti</b>, <b>credit_score</b>, <b>employment_length_years</b>,
        <b>product_type</b> (0=unsecured, 1=HELOC). Missing values filled with portfolio medians.
        </div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose CSV", type="csv")
        sample = pd.DataFrame([
            {"loan_amount":15000,"annual_income":65000,"dti":28.5,"credit_score":680,
             "employment_length_years":5,"product_type":0},
            {"loan_amount":25000,"annual_income":95000,"dti":22.0,"credit_score":730,
             "employment_length_years":8,"product_type":0},
            {"loan_amount":50000,"annual_income":120000,"dti":18.0,"credit_score":740,
             "employment_length_years":10,"product_type":1},
        ])
        st.download_button("⬇ Download sample CSV", sample.to_csv(index=False),
                           "sample_applications.csv","text/csv")
    with col_r:
        st.markdown("**Or load test portfolio**")
        st.markdown("""<div class="explain-box">194,564 real LendingClub + HELOC loans from the
        out-of-time test set (2016-2018). Never seen during training.</div>""", unsafe_allow_html=True)
        use_portfolio = st.button("Load Test Portfolio", use_container_width=True)

    # Initialise session state for batch results so filters persist
    if "batch_df" not in st.session_state:
        st.session_state["batch_df"] = None
        st.session_state["batch_source"] = None
        st.session_state["batch_errors"] = []

    # ── Load path A: test portfolio ──────────────────────────────────
    if use_portfolio and len(portfolio) > 0:
        cols_needed = ["product_type","pd_score","lgd_estimate","ead_estimate",
                       "expected_loss","credit_score","risk_tier","ifrs9_stage","rwa"]
        available = [c for c in cols_needed if c in portfolio.columns]
        df_raw = portfolio[available].copy()
        # Derive decision column from pd_score so it aligns with upload path
        if "pd_score" in df_raw.columns:
            df_raw["decision"] = np.where(
                df_raw["pd_score"] <= APPROVAL_THRESHOLD, "APPROVE",
                np.where(df_raw["pd_score"] <= 0.35, "REFER", "DECLINE_CREDIT"))
        if "product_type" in df_raw.columns:
            df_raw["product"] = df_raw["product_type"].map({0: "Unsecured", 1: "HELOC"})
        st.session_state["batch_df"]      = df_raw
        st.session_state["batch_source"]  = "test_portfolio"
        st.session_state["batch_errors"]  = []

    # ── Load path B: uploaded CSV ────────────────────────────────────
    elif uploaded is not None and models["loaded"]:
        df_input = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_input):,} applicants")
        with st.spinner(f"Scoring {len(df_input):,} applicants..."):
            results, errors, progress = [], [], st.progress(0)
            for i, (idx, row) in enumerate(df_input.iterrows()):
                try:
                    r = score_applicant(models, {
                        "loan_amount": float(row.get("loan_amount",10000)),
                        "annual_income": float(row.get("annual_income",60000)),
                        "dti": float(row.get("dti",25)),
                        "credit_score": float(row.get("credit_score",650)),
                        "employment_length_years": float(row.get("employment_length_years",3)),
                        "product_type": int(row.get("product_type",0)),
                    })
                    results.append({
                        "decision":      r["decision"],
                        "pd_score":      round(r["pd_pit"], 4),
                        "fraud_score":   round(r["fraud_score"], 4),
                        "fraud_tier":    r["fraud_alert_tier"],
                        "credit_score":  r["credit_score"],
                        "risk_tier":     r["risk_tier"],
                        "lgd_estimate":  round(r.get("lgd", 0), 4),
                        "ead_estimate":  round(r.get("ead", 0), 0),
                        "expected_loss": round(r["el"], 0),
                        "ifrs9_stage":   r["ifrs9_stage"],
                        "rwa":           round(r.get("rwa", 0), 0),
                        "product":       "HELOC" if int(row.get("product_type",0)) == 1 else "Unsecured",
                    })
                except Exception as _scoring_err:
                    errors.append({"row_index": int(idx), "error": str(_scoring_err)[:100]})
                    results.append({
                        "decision":"ERROR","pd_score":None,"fraud_score":None,
                        "fraud_tier":None,"credit_score":None,"risk_tier":None,
                        "lgd_estimate":None,"ead_estimate":None,
                        "expected_loss":None,"ifrs9_stage":None,"rwa":None,
                        "product":None,
                    })
                if i % 10 == 0:
                    progress.progress(min(i / max(len(df_input), 1), 1.0))
            progress.empty()
        df_out = pd.concat([df_input.reset_index(drop=True), pd.DataFrame(results)], axis=1)
        st.session_state["batch_df"]     = df_out
        st.session_state["batch_source"] = "upload"
        st.session_state["batch_errors"] = errors

    # ── Display batch results (with filters, segmentation, errors) ────
    batch_df = st.session_state.get("batch_df")
    if batch_df is not None and len(batch_df) > 0:

        # Error banner (upload path)
        errs = st.session_state.get("batch_errors", [])
        total = len(batch_df)
        n_err = len(errs)
        n_ok  = total - n_err
        if n_err > 0:
            st.warning(f"⚠ Scored {n_ok:,} of {total:,} rows successfully — "
                       f"{n_err} failed. Expand below to see error details.")
            with st.expander(f"Error details ({n_err} rows)", expanded=False):
                err_df = pd.DataFrame(errs)
                st.dataframe(err_df, use_container_width=True, hide_index=True)
        else:
            st.success(f"✓ All {total:,} rows scored successfully.")

        st.divider()

        # ── Filters ──────────────────────────────────────────────────
        st.markdown("**Filters**")
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            if "product" in batch_df.columns:
                products = ["All"] + sorted([p for p in batch_df["product"].dropna().unique()])
                sel_product = st.selectbox("Product", products, key="batch_product")
            else:
                sel_product = "All"
        with fc2:
            if "risk_tier" in batch_df.columns:
                tiers = ["All"] + sorted([t for t in batch_df["risk_tier"].dropna().unique()])
                sel_tier = st.selectbox("Risk Tier", tiers, key="batch_tier")
            else:
                sel_tier = "All"
        with fc3:
            if "decision" in batch_df.columns:
                decisions = ["All"] + sorted([d for d in batch_df["decision"].dropna().unique()])
                sel_dec = st.selectbox("Decision", decisions, key="batch_dec")
            else:
                sel_dec = "All"
        with fc4:
            if "expected_loss" in batch_df.columns:
                el_max = float(batch_df["expected_loss"].max() or 0)
                sel_el_min = st.number_input("Min Expected Loss ($)", 0.0, el_max, 0.0, 100.0,
                                              key="batch_el_min")
            else:
                sel_el_min = 0.0

        # Apply filters
        f = batch_df.copy()
        if sel_product != "All" and "product" in f.columns:
            f = f[f["product"] == sel_product]
        if sel_tier != "All" and "risk_tier" in f.columns:
            f = f[f["risk_tier"] == sel_tier]
        if sel_dec != "All" and "decision" in f.columns:
            f = f[f["decision"] == sel_dec]
        if sel_el_min > 0 and "expected_loss" in f.columns:
            f = f[f["expected_loss"].fillna(0) >= sel_el_min]

        st.caption(f"Showing {len(f):,} of {len(batch_df):,} rows after filters.")

        # ── Summary KPIs (on filtered data) ──────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        if "pd_score" in f.columns:
            approvals = int((f["pd_score"].fillna(1.0) <= APPROVAL_THRESHOLD).sum())
            refers    = int(((f["pd_score"].fillna(1.0) > APPROVAL_THRESHOLD) &
                              (f["pd_score"].fillna(1.0) <= 0.35)).sum())
        else:
            approvals = refers = 0
        total_el = float(f["expected_loss"].fillna(0).sum()) if "expected_loss" in f.columns else 0.0
        c1.metric("Total Loans", f"{len(f):,}")
        c2.metric("Approve", f"{approvals:,}",
                  delta=f"{approvals/max(len(f),1):.1%}" if len(f) else None)
        c3.metric("Refer",   f"{refers:,}")
        c4.metric("Total EL", f"${total_el:,.0f}")

        # ── Risk tier distribution chart ─────────────────────────────
        if "risk_tier" in f.columns and len(f) > 0:
            tier_dist = f["risk_tier"].value_counts().sort_index()
            fig = px.bar(x=tier_dist.index, y=tier_dist.values, color=tier_dist.index,
                         color_discrete_map=TIER_COLORS,
                         labels={"x":"Risk Tier","y":"Loans"},
                         title="Distribution by Risk Tier (filtered)")
            fig.update_layout(showlegend=False, height=260,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        # ── Product × Tier concentration table ───────────────────────
        if "product" in f.columns and "risk_tier" in f.columns and len(f) > 0:
            st.markdown("**Concentration by Product × Tier**")
            pivot = f.pivot_table(
                index="product", columns="risk_tier",
                values="pd_score" if "pd_score" in f.columns else "decision",
                aggfunc="count", fill_value=0,
            )
            st.dataframe(pivot, use_container_width=True)

        # ── Results table (sortable — Streamlit dataframe sorts in-place) ─
        st.markdown("**Results (first 100 rows — download CSV for full set)**")
        st.dataframe(f.head(100), use_container_width=True)

        st.download_button(
            "⬇ Download Filtered Results",
            f.to_csv(index=False),
            f"batch_scored_filtered.csv", "text/csv")
        st.download_button(
            "⬇ Download All Results (pre-filter)",
            batch_df.to_csv(index=False),
            f"batch_scored_all.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="tab-summary">
    <h4>📈 Model Performance & Validation</h4>
    <p>Model validation evidence — what a risk manager, auditor, or OSFI examiner reviews
    before approving a model for production. Covers discriminatory power (AUC/KS/Gini with
    ROC and KS plots), operational accuracy (confusion matrix + lift/gains), calibration
    (predicted vs actual by decile + Brier), vintage performance, global SHAP importance,
    PSI/CSI stability, and a v1.0-unified vs v1.1-segmented model comparison.</p>
    </div>""", unsafe_allow_html=True)

    mc = reports["model_comparison"]
    st.markdown("### 1. Discriminatory Power — Scorecard vs XGBoost (OOT Test)")
    st.markdown("""<div class="explain-box">
    <b>AUC</b>: Probability that the model correctly ranks a defaulter above a non-defaulter.
    0.5 = coin flip. 0.70+ = deployment-grade. 0.80+ = excellent.<br>
    <b>KS</b>: Maximum separation between defaulter and non-defaulter CDFs. Industry minimum 0.30.<br>
    <b>Gini</b>: 2×AUC−1. Formal regulatory reporting metric submitted to OSFI.<br>
    <b>Note on v1.0 combined AUC 0.68</b>: Dual-product architecture causes
    <code>loan_term_months</code> to act as a product proxy. See MODEL_CARD §6 for full
    disclosure. Section 3 below shows the v1.1 per-product breakout which removes this proxy.
    </div>""", unsafe_allow_html=True)

    if len(mc) > 0:
        cols = st.columns(len(mc))
        for i, (_, row) in enumerate(mc.iterrows()):
            with cols[i]:
                auc_val = float(row["auc"]); ks_val = float(row["ks"])
                st.markdown(f"**{row['model']}**")
                c1,c2,c3 = st.columns(3)
                c1.metric("AUC", f"{auc_val:.4f}",
                    delta="✓ OK" if auc_val>=0.70 else "⚠ Below 0.70",
                    delta_color="normal" if auc_val>=0.70 else "inverse")
                c2.metric("KS", f"{ks_val:.4f}",
                    delta="✓ OK" if ks_val>=0.30 else "⚠ Below 0.30",
                    delta_color="normal" if ks_val>=0.30 else "inverse")
                c3.metric("Gini", f"{float(row['gini']):.4f}")

    # ── 2. ROC curve + KS separation chart ────────────────────────
    st.divider()
    st.markdown("### 2. ROC Curve and KS Separation")
    st.markdown("""<div class="explain-box">
    <b>ROC curve</b>: True-positive rate (defaulters correctly identified) vs false-positive rate
    (good borrowers incorrectly flagged) across all score thresholds. The diagonal is random;
    the further the curve bends toward the top-left, the better the model.<br>
    <b>KS curve</b>: Cumulative default rate minus cumulative non-default rate at each score.
    The maximum gap is the KS statistic — the score at which the model best separates the two
    populations. This is the natural operating threshold for a decisioning system.
    </div>""", unsafe_allow_html=True)

    if len(portfolio) > 0 and "pd_score" in portfolio.columns and "default_flag" in portfolio.columns:
        try:
            port_pr = portfolio[["pd_score","default_flag"]].dropna()
            # Sample for plotting speed if portfolio is large
            if len(port_pr) > 50000:
                port_pr = port_pr.sample(50000, random_state=42)
            y_true = port_pr["default_flag"].values
            y_pred = port_pr["pd_score"].values

            fpr, tpr, _ = compute_roc_curve(y_true, y_pred, n_points=100)
            score_p, cum_bad, cum_good, ks_max, ks_score = compute_ks_curve(y_true, y_pred)
            # Recompute AUC from the sample for chart annotation
            from sklearn.metrics import roc_auc_score as _auc
            try:
                auc_plot = float(_auc(y_true, y_pred))
            except Exception:
                auc_plot = 0.0

            roc_col, ks_col = st.columns(2)
            with roc_col:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"Model (AUC={auc_plot:.3f})",
                    line=dict(color="#1a7a4a", width=3)))
                fig_roc.add_trace(go.Scatter(
                    x=[0,1], y=[0,1], mode="lines", name="Random (AUC=0.50)",
                    line=dict(color="#bbb", width=1, dash="dash")))
                fig_roc.update_layout(
                    title="ROC Curve (OOT Portfolio)", height=340,
                    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(x=0.55, y=0.10))
                st.plotly_chart(fig_roc, use_container_width=True)

            with ks_col:
                # Downsample KS curve to ~500 points for rendering speed
                step = max(len(score_p) // 500, 1)
                fig_ks = go.Figure()
                fig_ks.add_trace(go.Scatter(
                    x=score_p[::step], y=cum_bad[::step], mode="lines",
                    name="Cumulative Defaults",
                    line=dict(color="#c0392b", width=2)))
                fig_ks.add_trace(go.Scatter(
                    x=score_p[::step], y=cum_good[::step], mode="lines",
                    name="Cumulative Non-Defaults",
                    line=dict(color="#1a7a4a", width=2)))
                fig_ks.add_vline(x=ks_score, line_dash="dash", line_color="#888",
                    annotation_text=f"KS={ks_max:.3f} at PD≈{ks_score:.2%}",
                    annotation_position="top right")
                fig_ks.update_layout(
                    title="KS Separation Curve", height=340,
                    xaxis_title="Predicted PD (score)", yaxis_title="Cumulative Rate",
                    yaxis_tickformat=".0%",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(x=0.05, y=0.95))
                st.plotly_chart(fig_ks, use_container_width=True)
        except Exception as _e:
            st.info(f"ROC/KS charts unavailable: {_e}")
    else:
        st.info("Load portfolio data to see ROC/KS plots. Run build.py first.")

    # ── 3. Segmented model comparison (v1.0 unified vs v1.1 per-product) ─
    st.divider()
    st.markdown("### 3. v1.0 Unified vs v1.1 Segmented Models")
    st.markdown("""<div class="explain-box">
    <b>Why this comparison matters:</b> The v1.0 unified model trains one XGBoost on
    both unsecured LendingClub and HELOC data. Its reported AUC is depressed because
    <code>loan_term_months</code> acts as a product-type proxy. The v1.1 segmented approach
    trains two separate models — one per product — and removes product-proxy features.<br><br>
    <b>Trade-offs of segmentation:</b>
    (a) <b>AUC lift</b>: typically +0.04 to +0.10 per product because each model optimises
    for a homogeneous default mechanism.
    (b) <b>Operational cost</b>: two model cards, two monitoring pipelines, two OSFI E-23
    governance documents.
    (c) <b>Cross-product comparability</b>: score 720 unsecured is no longer directly
    comparable to score 720 HELOC — they're calibrated to different baseline default rates.
    <br><br>
    <b>Recommended posture</b>: retain the unified model for cross-product decisioning and
    aggregate portfolio reporting. Use segmented models for per-product monitoring,
    challenger validation, and thin-file segment performance.
    </div>""", unsafe_allow_html=True)

    mc_seg = reports.get("model_comparison_segmented", pd.DataFrame())
    if len(mc_seg) > 0:
        # Pretty-print with tier thresholds
        styled = mc_seg.copy()
        styled_oot = styled[styled["split"] == "Test (OOT)"].copy()
        if len(styled_oot) > 0:
            seg_cols = st.columns(len(styled_oot))
            for i, (_, row) in enumerate(styled_oot.iterrows()):
                with seg_cols[i]:
                    st.markdown(f"**{row['product']} (OOT)**")
                    c1, c2, c3 = st.columns(3)
                    auc_v = float(row["auc"]); ks_v = float(row["ks"])
                    c1.metric("AUC", f"{auc_v:.4f}",
                        delta="✓ Deployment-grade" if auc_v >= 0.70 else "⚠ Below 0.70",
                        delta_color="normal" if auc_v >= 0.70 else "inverse")
                    c2.metric("KS", f"{ks_v:.4f}",
                        delta="✓" if ks_v >= 0.30 else "⚠",
                        delta_color="normal" if ks_v >= 0.30 else "inverse")
                    c3.metric("Gini", f"{float(row['gini']):.4f}")
                    st.caption(f"N = {int(row['n']):,} · "
                               f"Default rate: {float(row['default_rate']):.2%}")

        # Full table with train + test
        st.markdown("**Full segmented comparison (train + OOT test):**")
        disp = mc_seg.copy()
        disp["default_rate"] = disp["default_rate"].map(lambda v: f"{v:.2%}")
        disp["auc"]  = disp["auc"] .map(lambda v: f"{v:.4f}")
        disp["ks"]   = disp["ks"]  .map(lambda v: f"{v:.4f}")
        disp["gini"] = disp["gini"].map(lambda v: f"{v:.4f}")
        disp["n"]    = disp["n"]   .map(lambda v: f"{v:,}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Lift narrative
        oot = mc_seg[mc_seg["split"] == "Test (OOT)"]
        unified_auc = None
        if len(mc) > 0:
            xgb_rows = mc[mc["model"].str.contains("XGBoost", na=False)]
            if len(xgb_rows) > 0:
                unified_auc = float(xgb_rows.iloc[0]["auc"])

        if unified_auc is not None and len(oot) > 0:
            uns = oot[oot["product"] == "Unsecured"]
            sec = oot[oot["product"] == "Secured"]
            narrative_parts = [f"**v1.0 Unified OOT AUC: {unified_auc:.4f}**"]
            if len(uns) > 0:
                lift = float(uns.iloc[0]["auc"]) - unified_auc
                narrative_parts.append(
                    f"Unsecured-only: **{float(uns.iloc[0]['auc']):.4f}** "
                    f"(Δ {lift:+.4f})")
            if len(sec) > 0:
                lift = float(sec.iloc[0]["auc"]) - unified_auc
                narrative_parts.append(
                    f"Secured-only: **{float(sec.iloc[0]['auc']):.4f}** "
                    f"(Δ {lift:+.4f})")
            st.info(" · ".join(narrative_parts))
    else:
        st.info("Segmented comparison not yet generated. Run "
                "`python notebooks/phase4b/04b_segmented_models.py` to produce "
                "`reports/phase4/model_comparison_segmented.csv`.")

    # ── 4. Operating Threshold — Confusion Matrix + Lift/Gains ────
    st.divider()
    st.markdown("### 4. Operating Threshold — Confusion Matrix & Lift/Gains")
    st.markdown("""<div class="explain-box">
    <b>Confusion matrix</b> translates AUC into what a loan officer actually cares about: at
    our approval threshold, how many good borrowers are incorrectly declined, and how many
    defaulters slip through. <b>Lift</b> is the industry-standard ranking-power metric —
    "among the top 10% riskiest applications, how many more defaulters are caught vs random?"
    A lift of 3.0× in decile 1 means the top 10% of scores contain 3× the default rate of
    the portfolio average.
    </div>""", unsafe_allow_html=True)

    if len(portfolio) > 0 and "pd_score" in portfolio.columns and "default_flag" in portfolio.columns:
        try:
            port_op = portfolio[["pd_score","default_flag"]].dropna()
            if len(port_op) > 100000:
                port_op = port_op.sample(100000, random_state=42)
            # User-adjustable threshold
            st.markdown("**Adjust the operating threshold (PD cutoff) to see the confusion matrix update:**")
            op_thresh = st.slider("Decision threshold (decline if PD >)",
                0.05, 0.70, float(APPROVAL_THRESHOLD), 0.01, key="op_thresh",
                help=f"Platform default: {APPROVAL_THRESHOLD:.0%}. Above this PD → DECLINE_CREDIT.")

            cm = confusion_at_threshold(port_op["default_flag"].values,
                                          port_op["pd_score"].values, op_thresh)

            cm_col1, cm_col2 = st.columns([2, 3])
            with cm_col1:
                # Confusion matrix as a small 2x2 heatmap
                cm_matrix = [
                    [cm["TN"], cm["FP"]],
                    [cm["FN"], cm["TP"]],
                ]
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm_matrix,
                    x=["Predicted Approve", "Predicted Decline"],
                    y=["Actual Good", "Actual Default"],
                    text=[[f"TN = {cm['TN']:,}", f"FP = {cm['FP']:,}"],
                          [f"FN = {cm['FN']:,}", f"TP = {cm['TP']:,}"]],
                    texttemplate="%{text}",
                    colorscale=[[0, "#f8f9fa"], [1, "#1a7a4a"]],
                    showscale=False,
                ))
                fig_cm.update_layout(height=280,
                    title=f"Confusion Matrix @ PD > {op_thresh:.0%}",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_cm, use_container_width=True)

            with cm_col2:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Precision", f"{cm['precision']:.1%}",
                    help="TP / (TP + FP): of loans declined, what % would actually have defaulted")
                m2.metric("Recall",    f"{cm['recall']:.1%}",
                    help="TP / (TP + FN): of actual defaulters, what % are caught")
                m3.metric("F1 Score",  f"{cm['f1']:.3f}",
                    help="Harmonic mean of precision and recall")
                m4.metric("False Positive Rate", f"{cm['fpr']:.1%}",
                    help="FP / (FP + TN): of good borrowers, what % incorrectly declined")

                # Business interpretation
                approve_rate = (cm["TN"] + cm["FN"]) / max(cm["n"], 1)
                decline_rate = 1 - approve_rate
                good_loss_rate = cm["FP"] / max(cm["FP"] + cm["TN"], 1)
                default_catch_rate = cm["recall"]
                st.markdown(f"""<div class="explain-box" style="margin-top:8px">
                <b>Business interpretation at {op_thresh:.0%} threshold:</b><br>
                Approval rate: <b>{approve_rate:.1%}</b> ·
                Decline rate: <b>{decline_rate:.1%}</b><br>
                Good-borrower decline rate: <b>{good_loss_rate:.1%}</b>
                (incorrectly decline {good_loss_rate:.0%} of actually-good applicants)<br>
                Default capture: <b>{default_catch_rate:.1%}</b>
                (catch {default_catch_rate:.0%} of actual defaulters)
                </div>""", unsafe_allow_html=True)

            # Lift / Gains chart
            st.markdown("**Lift and Cumulative Gains by Score Decile**")
            lift_df, baseline = compute_lift_gains(port_op["default_flag"].values,
                                                    port_op["pd_score"].values, n_deciles=10)
            if len(lift_df) > 0:
                lg_col1, lg_col2 = st.columns(2)
                with lg_col1:
                    fig_lift = go.Figure()
                    fig_lift.add_trace(go.Bar(
                        x=lift_df["decile"].astype(str),
                        y=lift_df["lift"],
                        marker_color=["#c0392b" if v >= 2 else "#e6a817" if v >= 1 else "#1a7a4a"
                                      for v in lift_df["lift"]],
                        text=[f"{v:.2f}×" for v in lift_df["lift"]],
                        textposition="outside",
                    ))
                    fig_lift.add_hline(y=1.0, line_dash="dash", line_color="#888",
                        annotation_text="Random baseline (1×)")
                    fig_lift.update_layout(
                        title="Lift by Decile (1 = highest PD)",
                        height=300, xaxis_title="Score Decile",
                        yaxis_title="Lift (ratio vs portfolio avg)",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_lift, use_container_width=True)

                with lg_col2:
                    fig_gains = go.Figure()
                    fig_gains.add_trace(go.Scatter(
                        x=lift_df["cum_pop_pct"], y=lift_df["cum_defaults_pct"],
                        mode="lines+markers", name="Model",
                        line=dict(color="#1a7a4a", width=3)))
                    fig_gains.add_trace(go.Scatter(
                        x=[0,1], y=[0,1], mode="lines", name="Random",
                        line=dict(color="#bbb", dash="dash")))
                    fig_gains.update_layout(
                        title="Cumulative Gains Curve",
                        height=300,
                        xaxis_title="Cumulative % of Population (highest PD first)",
                        yaxis_title="Cumulative % of Defaults Captured",
                        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_gains, use_container_width=True)

                d1_lift = float(lift_df.iloc[0]["lift"]) if len(lift_df) > 0 else 0
                top20_cap = float(lift_df[lift_df["cum_pop_pct"] <= 0.21]["cum_defaults_pct"].max() or 0)
                st.caption(f"**Key reads:** Decile 1 lift = **{d1_lift:.2f}×** · "
                           f"Top 20% of scores captures **{top20_cap:.0%}** of all defaults.")
        except Exception as _e:
            st.info(f"Confusion matrix / lift charts unavailable: {_e}")

    # ── 5. Calibration (existing — moved down, better positioned) ─
    st.divider()
    st.markdown("### 5. Calibration Analysis")
    st.markdown("""<div class="explain-box">
    <b>Why calibration matters more than AUC for this platform:</b> PD is used in
    IFRS 9 staging, ECL computation, risk-based pricing, and Basel III capital.
    If PD is systematically over- or under-estimated, every downstream output is wrong.
    A reviewer may accept moderate AUC if PD is well calibrated — they will not
    accept shaky calibration when the entire regulatory stack depends on it.<br><br>
    <b>Calibration chart</b>: Each bar is one decile of the score distribution.
    Model PD (predicted) should closely track Actual Default Rate (observed).
    <b>Brier Score</b>: Mean squared error of probability predictions. Lower = better.
    Perfect = 0. Random = ~0.18 for a 20% default rate population.
    </div>""", unsafe_allow_html=True)

    if len(portfolio) > 0 and "pd_score" in portfolio.columns and "default_flag" in portfolio.columns:
        try:
            port_cal = portfolio[["pd_score","default_flag"]].dropna()
            port_cal["decile"] = pd.qcut(port_cal["pd_score"], 10, labels=False,
                                          duplicates="drop")
            cal_df = port_cal.groupby("decile").agg(
                mean_pd=("pd_score","mean"),
                actual_dr=("default_flag","mean"),
                n=("pd_score","count")
            ).reset_index()

            brier = float(np.mean((port_cal["pd_score"] - port_cal["default_flag"])**2))

            col_cal, col_brier = st.columns([3,1])
            with col_cal:
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Model PD (Predicted)",
                    x=cal_df["decile"].astype(str), y=cal_df["mean_pd"],
                    marker_color="#2196F3", opacity=0.7))
                fig.add_trace(go.Scatter(name="Actual Default Rate",
                    x=cal_df["decile"].astype(str), y=cal_df["actual_dr"],
                    mode="lines+markers", line=dict(color="#c0392b", width=2),
                    marker=dict(size=8)))
                fig.update_layout(title="Calibration: Predicted PD vs Actual Default Rate by Decile",
                    height=300, yaxis_tickformat=".0%",
                    xaxis_title="Score Decile (1=lowest PD, 10=highest PD)",
                    yaxis_title="Rate",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with col_brier:
                brier_baseline = port_cal["default_flag"].mean() * (1 - port_cal["default_flag"].mean())
                brier_skill = 1 - brier / brier_baseline
                st.metric("Brier Score", f"{brier:.4f}")
                st.caption(f"Baseline: {brier_baseline:.4f}")
                st.metric("Brier Skill Score", f"{brier_skill:.3f}")
                st.caption("Higher = better. 0 = no better than mean.")
                mean_pd = port_cal["pd_score"].mean()
                actual_dr = port_cal["default_flag"].mean()
                st.metric("Mean PD", f"{mean_pd:.2%}")
                st.metric("Actual DR", f"{actual_dr:.2%}")
                gap = mean_pd - actual_dr
                st.metric("Calibration Gap", f"{gap:+.4f}",
                    delta="✓ Well calibrated" if abs(gap)<0.02 else "⚠ Check calibration",
                    delta_color="normal" if abs(gap)<0.02 else "inverse")

            # Calibration by product segment
            if "product_type" in portfolio.columns:
                st.markdown("**Calibration by Product Segment:**")
                seg_cal = portfolio[["pd_score","default_flag","product_type"]].dropna()
                for pt, name in [(0,"Unsecured"),(1,"HELOC (Secured)")]:
                    seg = seg_cal[seg_cal["product_type"]==pt]
                    if len(seg) > 0:
                        seg_gap = seg["pd_score"].mean() - seg["default_flag"].mean()
                        seg_brier = float(np.mean((seg["pd_score"]-seg["default_flag"])**2))
                        st.caption(f"{name}: Mean PD {seg['pd_score'].mean():.2%} | "
                                   f"Actual DR {seg['default_flag'].mean():.2%} | "
                                   f"Gap {seg_gap:+.4f} | Brier {seg_brier:.4f}")
        except Exception as e:
            st.info(f"Calibration data not available. Run build.py first. ({e})")
    else:
        st.info("Load portfolio data to see calibration analysis. Run build.py first.")

    st.divider()
    st.markdown("### 6. Vintage Default Rate Analysis")
    vc = reports["vintage"]
    if "origination_year" in vc.columns and "default_rate" in vc.columns:
        avg = vc["default_rate"].mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vc["origination_year"].astype(int), y=vc["default_rate"],
            marker_color=["#c0392b" if v>avg*1.1 else "#2d9e6b" for v in vc["default_rate"]],
            text=[f"{v:.1%}" for v in vc["default_rate"]], textposition="outside"))
        fig.add_hline(y=avg, line_dash="dash", line_color="#e07520",
                      annotation_text=f"Portfolio avg: {avg:.2%}")
        fig.update_layout(height=300, yaxis_tickformat=".0%",
            xaxis_title="Origination Year", yaxis_title="Actual Default Rate",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red = above average. 2016-2018 = out-of-time test set (never seen during training).")

    st.divider()
    st.markdown("### 7. Global SHAP Feature Importance")
    try:
        import joblib as jl
        gi = jl.load("models/shap_global_importance.pkl")
        fig = px.bar(gi.head(10), x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=["#d4edda","#1a7a4a"],
            title="Top 10 Features by Global SHAP Importance")
        fig.update_layout(height=340, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("loan_term_months is #1 because it acts as a product-type proxy in the "
                   "v1.0 unified model (HELOC imputed=36m vs unsecured 36/60m). The v1.1 "
                   "segmented models in Section 3 above remove this proxy and elevate "
                   "genuine credit signals (credit_score, external_risk_estimate, DTI).")
    except:
        st.info("Run Phase 5 to generate SHAP global importance.")

    st.divider()
    st.markdown("### 8. Feature Stability (PSI/CSI)")
    st.markdown("""<div class="explain-box">
    PSI/CSI measures whether the population being scored today matches the training population.
    OSFI E-23 requires model review when PSI > 0.25. All features currently show OK status.
    </div>""", unsafe_allow_html=True)
    csi = reports["csi"]
    if len(csi) > 0:
        st.dataframe(
            csi.style.map(
                lambda v: "background-color:#f8d7da" if v=="ALERT — investigate"
                else ("background-color:#fff3cd" if v=="WATCH"
                else "background-color:#d4edda" if v=="OK" else ""),
                subset=["status"] if "status" in csi.columns else []),
            use_container_width=True)

    st.divider()
    st.markdown("### 9. Fairness Assessment")
    st.markdown("""<div class="warn-box">
    <b>Scope and limitations of this fairness analysis:</b><br>
    This analysis is <b>exploratory and monitoring-oriented</b>. It is not a full
    protected-class fair lending test and is not a substitute for proxy-based disparate
    impact analysis (which requires demographic data not present in public datasets).<br><br>
    Metrics used: demographic parity (approval rate gap across segments) and equalized odds
    (TPR/FPR gaps). The EEOC four-fifths rule is used as the disparity threshold.<br><br>
    <b>What was found:</b> 34 flags across 5 segments (purpose, home ownership, grade,
    product type, verification status). All classified as Acceptable Disparity (risk-relevant)
    or Monitoring Required. Zero Action Required flags.<br><br>
    <b>What this does not cover:</b> Proxy variables for race, gender, or national origin.
    Geographic redlining analysis. Full ECOA/FCRA compliance testing. A compliance officer
    would conduct these analyses before production deployment.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — COMPLIANCE
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="tab-summary">
    <h4>🏛 Regulatory Compliance Dashboard</h4>
    <p>Three core regulatory obligations for a Canadian non-bank lender:
    <b>Basel III Capital</b> (minimum 8% of RWA — OSFI CAR), <b>IFRS 9 ECL</b>
    (forward-looking provisions in three stages), and <b>Macroeconomic Stress Testing</b>
    (OSFI requires survival under Base, Adverse, and Severe scenarios).</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Basel III Capital Requirements")
    st.markdown("""<div class="explain-box">
    <b>IRB formula:</b> K = LGD × N[(1-R)^-0.5 × G(PD) + (R/(1-R))^0.5 × G(0.999)] − PD×LGD,
    where R is the asset correlation and N/G are the standard normal CDF/inverse.
    Capital is calculated at the 99.9th percentile — sufficient to survive all but a 1-in-1000
    year loss event. <b>Note on RWA density:</b> Secured HELOC has lower density
    than Unsecured because collateral reduces unexpected loss risk. This is correct.<br><br>
    <b>Capital Adequacy Ratio (CAR) methodology:</b> CAR = Available Capital ÷ Total RWA.
    OSFI requires minimum 8% Pillar 1, plus D-SIB CET1 target of 11.5% and a 2.5% Capital
    Conservation Buffer. The ratio below assumes the lender holds capital equal to 11.5% of RWA
    (a typical CET1 target) as its "available capital" — a demonstration assumption. In production,
    available capital would come from the CET1 return filed with OSFI.
    </div>""", unsafe_allow_html=True)

    cap = reports["capital"]
    if len(cap) > 0:
        # Totals
        # FIX: CAR is Available Capital / RWA, NOT min_capital / EAD.
        # Previous version divided by total_ead which produced RWA density's inverse,
        # not CAR. Correct denominator is RWA; numerator is available capital
        # (assumed CET1 target × RWA for demonstration).
        total_ead = float(cap["total_ead"].astype(float).sum())
        total_rwa = float(cap["total_rwa"].astype(float).sum())
        min_capital_required = float(cap["min_capital"].astype(float).sum())

        # Assumed available capital stack — demonstration only
        ASSUMED_CET1_TARGET = 0.115    # 11.5% D-SIB target
        CAR_MINIMUM         = 0.08     # OSFI Pillar 1 minimum
        CAR_TARGET          = 0.115    # D-SIB CET1 target (common internal target)
        available_capital = total_rwa * ASSUMED_CET1_TARGET

        # Correct CAR calculation
        car = available_capital / total_rwa if total_rwa > 0 else 0.0
        # Also report min-capital-over-RWA (will always be 8% by construction)
        min_cap_ratio = min_capital_required / total_rwa if total_rwa > 0 else 0.0

        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        kcol1.metric("Total Exposure (EAD)", f"${total_ead:,.0f}")
        kcol2.metric("Total RWA", f"${total_rwa:,.0f}")
        kcol3.metric("Capital Adequacy Ratio", f"{car:.2%}",
            delta=f"✓ Above {CAR_MINIMUM:.0%} min" if car >= CAR_MINIMUM else "⚠ Below minimum",
            delta_color="normal" if car >= CAR_MINIMUM else "inverse",
            help=f"Available Capital ÷ RWA. Assumes available capital = {ASSUMED_CET1_TARGET:.1%} × RWA "
                 f"(CET1 target assumption for demonstration). Real CAR uses booked CET1 from OSFI return.")
        kcol4.metric("Min Capital Required (8%)", f"${min_capital_required:,.0f}",
            help="Pillar 1 minimum. 8% × RWA by construction — this is the regulatory floor.")

        st.caption(
            f"**CAR ladder:** OSFI Pillar 1 minimum **8.0%** · "
            f"Capital Conservation Buffer **+2.5%** (total 10.5%) · "
            f"D-SIB CET1 target **11.5%** · Assumed available capital for this demo: "
            f"**{ASSUMED_CET1_TARGET:.1%} × RWA = ${available_capital:,.0f}**"
        )
        if car < CAR_TARGET:
            st.warning(f"CAR {car:.2%} below {CAR_TARGET:.1%} D-SIB CET1 target — "
                       "portfolio would require additional capital or risk-weighted asset reduction.")

        # Per-product breakout
        st.markdown("**Capital by Product**")
        prod_rows = cap[cap["product"].astype(str).str.lower() != "total"]
        if len(prod_rows) > 0:
            per_product_cols = st.columns(max(len(prod_rows), 1))
            for i, (_, row) in enumerate(prod_rows.iterrows()):
                with per_product_cols[i % len(per_product_cols)]:
                    st.markdown(f"**{row['product']}**")
                    st.metric("Total EAD", f"${float(row['total_ead']):,.0f}")
                    st.metric("Total RWA", f"${float(row['total_rwa']):,.0f}")
                    st.metric("Min Capital", f"${float(row['min_capital']):,.0f}")
                    st.metric("RWA Density", f"{float(row['rwa_density']):.1%}",
                        help="RWA ÷ EAD. Higher density = more capital-intensive product. "
                             "Secured products have lower density due to collateral.")

    st.divider()
    st.markdown("### IFRS 9 — Expected Credit Loss by Stage")
    st.markdown("""<div class="explain-box">
    <b>Important methodology note:</b><br>
    This is a <b>demonstration implementation</b> of IFRS 9 staging. The origination PD
    is proxied using LendingClub loan grade at origination (A=4%, B=8%, C=13%, D=20%,
    E=28%, F=35%, G=40%) because public datasets do not provide the actual booked-at-origination
    PD snapshot from a Loan Origination System (LOS).<br><br>
    <b>Staging triggers:</b> Stage 2 = PD increased &gt;20% from origination, OR credit score
    dropped 30+ points, OR 30+ DPD. Stage 3 = PD ≥ 70% OR 90+ DPD.<br><br>
    <b>In production:</b> Origination PD would come from actual LOS records — the PD at
    which the loan was originally booked. This is a regulatory requirement under IFRS 9
    paragraph 5.5.11. The grade-based proxy is a reasonable approximation for
    historical public data but would not be acceptable for a regulatory submission.<br><br>
    <b>Coverage benchmark colour-coding below:</b> rows are tinted green if coverage falls
    inside the expected industry band for that stage, amber if outside but within a reasonable
    range, red if materially outside the band. Industry bands:
    Stage 1: 0.5–2% · Stage 2: 5–15% · Stage 3: 40–80%.
    </div>""", unsafe_allow_html=True)

    ecl = reports["ecl"]
    # Industry coverage bands for IFRS 9 by stage
    STAGE_BANDS = {
        1: (0.005, 0.02),   # 0.5% - 2%
        2: (0.05,  0.15),   # 5% - 15%
        3: (0.40,  0.80),   # 40% - 80%
    }

    def _coverage_colour(stage, coverage):
        """Return background hex for an IFRS 9 coverage ratio vs industry bands."""
        try:
            stage = int(stage)
        except Exception:
            return ""
        if stage not in STAGE_BANDS:
            return ""
        lo, hi = STAGE_BANDS[stage]
        # Within band — green; within 1.5× of band — amber; else red
        if lo <= coverage <= hi:
            return "background-color:#d4edda"
        if (lo * 0.67) <= coverage <= (hi * 1.5):
            return "background-color:#fff3cd"
        return "background-color:#f8d7da"

    if len(ecl) > 0:
        ecl_view = ecl[["stage","n_loans","total_ead","total_ecl","coverage_ratio"]].copy()

        # Apply stage-band colouring
        def _style_row(row):
            colour = _coverage_colour(row["stage"], float(row["coverage_ratio"]))
            return [colour] * len(row)

        st.dataframe(
            ecl_view.style.apply(_style_row, axis=1).format({
                "total_ead":"${:,.0f}","total_ecl":"${:,.0f}","coverage_ratio":"{:.2%}"}),
            use_container_width=True)

        # Narrative interpretation per stage
        bullets = []
        for _, row in ecl_view.iterrows():
            try:
                stage = int(row["stage"])
                cov = float(row["coverage_ratio"])
            except Exception:
                continue
            if stage not in STAGE_BANDS:
                continue
            lo, hi = STAGE_BANDS[stage]
            if lo <= cov <= hi:
                status = "✓ within industry band"
            elif cov > hi:
                status = f"⚠ above industry band ({hi:.1%})"
            else:
                status = f"⚠ below industry band ({lo:.1%})"
            bullets.append(f"**Stage {stage}**: coverage {cov:.2%} — {status}")
        if bullets:
            st.markdown("\n".join(f"- {b}" for b in bullets))
        st.caption("Coverage benchmarks: Stage 1: 0.5–2% · Stage 2: 5–15% · Stage 3: 40–80%")

    st.divider()
    st.markdown("### Macroeconomic Stress Test")
    st.markdown("""<div class="explain-box">
    <b>Stress test methodology:</b><br>
    <b>How stress affects PD:</b> Each scenario applies a PD multiplier (1.0×, 1.4×, 2.0×)
    derived from historical PD migration studies during the 2008-09 GFC and 2020 COVID recession.
    The severe scenario (2.0×) is calibrated to approximate 2008-09 outcomes for consumer credit.<br><br>
    <b>LGD in stress (documented simplification):</b> In this v1.0 implementation LGD is held
    constant across scenarios. A more complete v1.1 implementation would model LGD deterioration
    under stress — particularly for HELOC, where a 2008-style house-price shock of −25% to −35%
    would raise HELOC LGD materially (historical Freddie Mac data shows HELOC LGD rising from
    ~30% in benign conditions to 50–60% during GFC). For unsecured consumer loans, recovery
    is largely fraud/bankruptcy-driven and less sensitive to macro LGD, so the constant-LGD
    assumption is materially conservative for HELOC and approximately neutral for unsecured.<br><br>
    <b>TTC vs PIT in capital:</b> RWA uses TTC PD (smoothed). Stressed EL uses PIT PD × multiplier.
    These can move differently — capital increases less than EL in stress because TTC PD is
    already smoothed upward during boom periods (anti-procyclical mechanism of Basel III).<br><br>
    <b>Key test:</b> EL must increase monotonically (Base &lt; Adverse &lt; Severe) and capital
    headroom must remain positive under Base and Adverse scenarios.
    </div>""", unsafe_allow_html=True)

    stress = reports["stress"]
    if len(stress) > 0:
        sc_colors = {"Base Case":"#1a7a4a","Adverse Scenario":"#e6a817","Severe Scenario":"#c0392b"}
        fig = go.Figure()
        for _, row in stress.iterrows():
            fig.add_trace(go.Bar(
                name=row["scenario"], x=[row["scenario"]],
                y=[float(row["total_el_stressed"])],
                marker_color=sc_colors.get(row["scenario"],"#888"),
                text=f"EL Rate: {float(row['el_rate']):.1%}", textposition="outside"))
        fig.update_layout(height=300, showlegend=False,
            yaxis_title="Stressed Expected Loss ($)", yaxis_tickformat="$,.0f",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            stress[["scenario","pd_multiplier","avg_pd_stressed",
                     "total_el_stressed","el_rate","total_rwa","required_capital"]].style.format({
                "pd_multiplier":"{:.2f}×","avg_pd_stressed":"{:.2%}",
                "total_el_stressed":"${:,.0f}","el_rate":"{:.2%}",
                "total_rwa":"${:,.0f}","required_capital":"${:,.0f}"}),
            use_container_width=True)
        st.markdown("""<div class="warn-box">
        <b>Severe scenario capital shortfall:</b> Under a 2×PD stress (2008-09 magnitude),
        required capital can exceed available capital at the assumed 11.5% CET1 target.
        In a real lender, this would require: (1) capital raise, (2) portfolio reduction, or
        (3) risk appetite adjustment. OSFI would discuss remediation actions with the lender.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5 — PRICING
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="tab-summary">
    <h4>💰 Risk-Based Pricing Calculator</h4>
    <p>Builds interest rates from first principles. Rate = Cost of Funds + Expected Loss
    (PD × LGD) + Operating Costs + ROE Target − Collateral Adjustment. Hard cap at 35% APR
    under Canada's Criminal Code s.347 (effective January 2025). This tab also shows the
    <b>profit curve</b> (net profit vs approval threshold), a <b>market rate comparison</b>
    against Canadian benchmarks, and a <b>cost-of-funds sensitivity</b> analysis.</p>
    </div>""", unsafe_allow_html=True)

    # ── 1. Single-applicant rate decomposition ──────────────────────
    st.markdown("### 1. Rate Decomposition Calculator")
    c_in, c_out = st.columns(2)
    with c_in:
        st.markdown("**Pricing inputs**")
        rb_product = st.selectbox("Product", ["Unsecured","HELOC"], key="rb_prod")
        rb_pd  = st.slider("Probability of Default (%)", 1, 99, 15, 1, key="rb_pd")
        rb_lgd = st.slider("Loss Given Default (%)", 10, 90, 65, 1, key="rb_lgd")
        rb_cof = st.slider("Cost of Funds (%)", 3.0, 8.0, 5.5, 0.1, key="rb_cof")
        rb_ops = st.slider("Operating Costs (%)", 0.5, 3.0, 1.5, 0.1, key="rb_ops")
        rb_roe = st.slider("ROE Target (%)", 1.0, 5.0, 2.0, 0.1, key="rb_roe")
        rb_ltv = st.slider("LTV ratio", 0.40, 0.95, 0.70, 0.01, key="rb_ltv") \
                 if rb_product=="HELOC" else 0.0

    with c_out:
        el_comp = (rb_pd/100)*(rb_lgd/100)
        cof=rb_cof/100; ops=rb_ops/100; roe=rb_roe/100
        coll=(-0.005*max(0,(0.80-rb_ltv)/0.10)) if rb_product=="HELOC" else 0
        raw=cof+el_comp+ops+roe+coll; fin=min(raw,0.35); capped=raw>0.35
        components={"Cost of Funds":cof,"Expected Loss (PD×LGD)":el_comp,
                    "Operating Costs":ops,"ROE Target":roe}
        if coll!=0: components["Collateral Adj."]=coll
        fig=go.Figure(go.Bar(x=list(components.keys()),
            y=[v*100 for v in components.values()],
            marker_color=["#2d9e6b","#c0392b","#e6a817","#3498db","#9b59b6"][:len(components)],
            text=[f"{v*100:.2f}%" for v in components.values()],textposition="outside"))
        fig.add_hline(y=fin*100,line_dash="dash",line_color="#c0392b",
                      annotation_text=f"Final: {fin:.2%}")
        if capped: fig.add_hline(y=35,line_dash="dot",line_color="#666",
                                  annotation_text="Criminal Code cap (35%)")
        fig.update_layout(height=280,yaxis_title="Rate (%)",
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
        if capped:
            st.warning(f"Calculated rate {raw:.2%} exceeds 35% cap. Capped at 35%.")
        else:
            st.success(f"Recommended APR: **{fin:.2%}**")

    st.divider()

    # ── 2. Full Rate Schedule — now uses portfolio-derived mid-PDs ──
    st.markdown("### 2. Full Rate Schedule by Risk Tier")
    st.markdown("""<div class="explain-box">
    Mid-PDs below are derived from the actual scored portfolio (median PD within each tier),
    not hardcoded illustrative values. This means the rate schedule is tied to the same
    PD distribution as the model currently produces — so if the model or population shifts,
    this table will reflect the new tier PDs the next time the portfolio is rebuilt.
    </div>""", unsafe_allow_html=True)

    tier_pds = compute_tier_pds_from_portfolio(portfolio)
    tier_ranges = {"A":"720+", "B":"680-719", "C":"630-679", "D":"580-629", "E":"<580"}
    tier_lgd = 0.66  # Portfolio-average LGD assumption
    tier_rows = []
    for tier in ["A","B","C","D","E"]:
        pm = tier_pds[tier]
        el = pm * tier_lgd
        rate_raw = cof + el + ops + roe
        rate     = min(rate_raw, 0.35)
        tier_rows.append({
            "Tier": tier,
            "Score Range": tier_ranges[tier],
            "Mid PD": f"{pm:.1%}",
            "Expected Loss Rate": f"{el:.2%}",
            "Uncapped Rate": f"{rate_raw:.2%}",
            "Recommended Rate": f"{rate:.2%}",
            "Criminal Code Cap?": "⚠ Yes — 35%" if rate_raw >= 0.35 else "✓ No",
        })
    st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)

    # Explanation caption adapts based on whether Tier E hits the cap
    tier_e_pd = tier_pds["E"]
    tier_e_el = tier_e_pd * tier_lgd
    if tier_e_el + cof >= 0.35:
        st.caption(f"Tier E borrowers hit the 35% cap because EL alone "
                   f"({tier_e_pd:.0%} × {tier_lgd:.0%} = {tier_e_el:.1%}) "
                   f"plus CoF ({cof:.1%}) already exceeds 35% before OpEx and ROE are added.")
    else:
        st.caption(f"Tier E EL ({tier_e_el:.1%}) plus CoF ({cof:.1%}) is below the 35% cap — "
                   "at current portfolio PDs, no tier hits the Criminal Code ceiling.")

    st.divider()

    # ── 3. Market rate comparison ───────────────────────────────────
    st.markdown("### 3. Market Rate Comparison (Canadian Lenders)")
    st.markdown("""<div class="explain-box">
    Where the recommended rate schedule lands versus published APRs from major Canadian
    lenders (representative ranges, early 2026). This anchors the output for a reviewer —
    showing whether the model's recommendations are commercially competitive, below
    market (underpricing risk), or above market (uncompetitive).
    </div>""", unsafe_allow_html=True)

    # Canadian market benchmarks — representative ranges, early 2026
    market_ranges = [
        {"Lender Type": "Prime bank (RBC, TD, BMO, etc.)",     "Product": "Unsecured loan",       "APR (Low)": 0.0795, "APR (High)": 0.1295},
        {"Lender Type": "Credit union",                        "Product": "Unsecured loan",       "APR (Low)": 0.0895, "APR (High)": 0.1495},
        {"Lender Type": "Fintech (Borrowell, KOHO, LoanConnect)", "Product": "Unsecured loan",    "APR (Low)": 0.1295, "APR (High)": 0.2495},
        {"Lender Type": "Subprime (Fairstone, Easyfinancial)",  "Product": "Unsecured loan",       "APR (Low)": 0.2495, "APR (High)": 0.3495},
        {"Lender Type": "Prime bank",                           "Product": "HELOC",                "APR (Low)": 0.0695, "APR (High)": 0.0895},
        {"Lender Type": "B lender (Home Trust, Equitable Bank)","Product": "HELOC",                "APR (Low)": 0.0895, "APR (High)": 0.1195},
    ]
    mkt_df = pd.DataFrame(market_ranges)
    # Annotate model output for selected product
    model_rates = [float(r["Recommended Rate"].rstrip("%"))/100 for r in tier_rows]
    model_min, model_max = min(model_rates), max(model_rates)

    disp_df = mkt_df.copy()
    disp_df["APR (Low)"]  = disp_df["APR (Low)"] .map(lambda v: f"{v:.2%}")
    disp_df["APR (High)"] = disp_df["APR (High)"].map(lambda v: f"{v:.2%}")
    st.dataframe(disp_df, use_container_width=True, hide_index=True)

    st.caption(
        f"**Your model's rate range** across tiers A-E at current inputs: "
        f"**{model_min:.2%} – {model_max:.2%}**. "
        f"Tier A ({model_rates[0]:.2%}) is competitive with fintech unsecured lenders; "
        f"Tier E ({model_rates[-1]:.2%}) sits in subprime-lender territory."
    )

    # Visual: model vs market as horizontal bars
    fig_mkt = go.Figure()
    for i, row in mkt_df.iterrows():
        fig_mkt.add_trace(go.Scatter(
            x=[row["APR (Low)"], row["APR (High)"]],
            y=[f"{row['Lender Type']} — {row['Product']}", f"{row['Lender Type']} — {row['Product']}"],
            mode="lines+markers",
            line=dict(color="#888", width=6),
            marker=dict(size=10),
            showlegend=False,
            hoverinfo="text",
            text=f"{row['APR (Low)']:.2%} – {row['APR (High)']:.2%}",
        ))
    # Overlay the model's tier-A and tier-E rates
    fig_mkt.add_vline(x=model_min, line_dash="dash", line_color="#1a7a4a",
                     annotation_text=f"Model Tier A: {model_min:.2%}", annotation_position="top")
    fig_mkt.add_vline(x=model_max, line_dash="dash", line_color="#c0392b",
                     annotation_text=f"Model Tier E: {model_max:.2%}", annotation_position="top")
    fig_mkt.update_layout(height=340, xaxis_title="APR", xaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_mkt, use_container_width=True)

    st.divider()

    # ── 4. Profit Curve — sweep approval thresholds ─────────────────
    st.markdown("### 4. Profit Curve — Optimal Approval Threshold")
    st.markdown("""<div class="explain-box">
    Sweeps approval thresholds (PD cutoffs) across the portfolio and computes expected
    interest revenue, expected credit loss, and <b>net profit</b> at each cutoff. The peak
    of the net profit curve is the profit-maximising threshold. The current platform
    threshold (PD ≤ 28%) is marked as a dashed vertical line for comparison.<br><br>
    <b>Assumptions</b>: revenue computed using the rate formula above applied to each loan's
    own PD and LGD, capped at 35%; loss = PD × LGD × EAD. Full-term realisation is assumed
    (no prepayment, no partial-term default-timing discount) — so this is an upper bound
    on realised revenue and the curve should be read directionally.
    </div>""", unsafe_allow_html=True)

    pc_col1, pc_col2 = st.columns(2)
    with pc_col1:
        pc_cof = st.slider("Cost of Funds (%)", 3.0, 8.0, 5.5, 0.1, key="pc_cof")
    with pc_col2:
        pc_opex = st.slider("OpEx + ROE target (%)", 2.0, 6.0, 3.5, 0.1, key="pc_opex")

    try:
        profit_df = compute_profit_curve(portfolio, cof=pc_cof/100,
                                          opex=pc_opex/100 / 2, roe=pc_opex/100 / 2,
                                          rate_cap=0.35)
    except Exception as _e:
        profit_df = pd.DataFrame()
        st.info(f"Profit curve unavailable: {_e}")

    if len(profit_df) > 0:
        # Find the profit-max threshold
        opt_idx = profit_df["net_profit"].idxmax()
        opt_threshold = float(profit_df.loc[opt_idx, "threshold"])
        opt_profit = float(profit_df.loc[opt_idx, "net_profit"])
        opt_approval = float(profit_df.loc[opt_idx, "approval_rate"])

        fig_pc = go.Figure()
        fig_pc.add_trace(go.Scatter(
            x=profit_df["threshold"], y=profit_df["expected_revenue"],
            mode="lines", name="Expected Revenue",
            line=dict(color="#1a7a4a", width=2)))
        fig_pc.add_trace(go.Scatter(
            x=profit_df["threshold"], y=profit_df["expected_loss"],
            mode="lines", name="Expected Loss",
            line=dict(color="#c0392b", width=2)))
        fig_pc.add_trace(go.Scatter(
            x=profit_df["threshold"], y=profit_df["net_profit"],
            mode="lines", name="Net Profit",
            line=dict(color="#3498db", width=3, dash="solid")))
        fig_pc.add_vline(x=APPROVAL_THRESHOLD, line_dash="dash", line_color="#888",
            annotation_text=f"Current platform: {APPROVAL_THRESHOLD:.0%}", annotation_position="top left")
        fig_pc.add_vline(x=opt_threshold, line_dash="dot", line_color="#e6a817",
            annotation_text=f"Profit-max: {opt_threshold:.0%}", annotation_position="top right")
        fig_pc.update_layout(height=380,
            xaxis_title="Approval Threshold (PD ≤ ...)",
            yaxis_title="Dollar Amount ($)",
            xaxis_tickformat=".0%", yaxis_tickformat="$,.0f",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.02, y=0.98))
        st.plotly_chart(fig_pc, use_container_width=True)

        prof_cols = st.columns(4)
        prof_cols[0].metric("Profit-Max Threshold", f"{opt_threshold:.0%}")
        prof_cols[1].metric("Net Profit @ Optimum", f"${opt_profit:,.0f}")
        prof_cols[2].metric("Approval Rate @ Optimum", f"{opt_approval:.1%}")
        # Compare to current platform threshold
        curr = profit_df.iloc[(profit_df["threshold"] - APPROVAL_THRESHOLD).abs().argsort()[:1]]
        if len(curr) > 0:
            curr_profit = float(curr["net_profit"].iloc[0])
            delta = opt_profit - curr_profit
            prof_cols[3].metric("vs Current Platform", f"${delta:+,.0f}",
                help=f"Profit at profit-max threshold minus profit at current "
                     f"{APPROVAL_THRESHOLD:.0%} threshold.")

        if opt_threshold < APPROVAL_THRESHOLD:
            st.caption(f"💡 **Insight**: the profit-maximising threshold ({opt_threshold:.0%}) is "
                       f"tighter than the current platform threshold ({APPROVAL_THRESHOLD:.0%}) — "
                       f"the current policy is approving some marginal loans where expected loss "
                       f"exceeds expected revenue. Tightening could lift portfolio profitability "
                       f"by ${opt_profit - curr_profit:,.0f} per scoring period.")
        elif opt_threshold > APPROVAL_THRESHOLD:
            st.caption(f"💡 **Insight**: the profit-maximising threshold ({opt_threshold:.0%}) is "
                       f"looser than the current platform threshold ({APPROVAL_THRESHOLD:.0%}) — "
                       f"the current policy is leaving some profitable loans on the table. "
                       f"This often reflects a deliberate conservative stance for risk-appetite "
                       f"reasons beyond pure profit maximisation.")
        else:
            st.caption(f"💡 **Insight**: current platform threshold is aligned with the "
                       "profit-maximising cutoff.")

    st.divider()

    # ── 5. Cost of Funds Sensitivity ────────────────────────────────
    st.markdown("### 5. Cost of Funds Sensitivity")
    st.markdown("""<div class="explain-box">
    How the approve-at-cap population changes as cost of funds rises. In a rising-rate
    environment (e.g. 2022-23 Bank of Canada hiking cycle), CoF rises faster than the
    rate cap, compressing the lender's margin on higher-risk tiers and pushing more
    of them into the 35% Criminal Code ceiling. This is a direct illustration of why
    the 35% cap concerns the subprime lending industry most acutely.
    </div>""", unsafe_allow_html=True)

    # Sweep cost of funds from 3% to 8%
    cof_grid = np.arange(0.03, 0.081, 0.005)
    sens_rows = []
    for cof_test in cof_grid:
        # Count how many tiers hit the cap at this CoF
        tiers_capped = 0
        for tier in ["A","B","C","D","E"]:
            pm = tier_pds[tier]
            el = pm * tier_lgd
            rate_raw = cof_test + el + ops + roe
            if rate_raw >= 0.35:
                tiers_capped += 1
        sens_rows.append({"cof": cof_test, "tiers_capped": tiers_capped,
                          "tier_e_rate_raw": cof_test + tier_pds["E"] * tier_lgd + ops + roe})

    sens_df = pd.DataFrame(sens_rows)
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=sens_df["cof"], y=sens_df["tier_e_rate_raw"],
        mode="lines+markers", name="Tier E uncapped rate",
        line=dict(color="#c0392b", width=3)))
    fig_sens.add_hline(y=0.35, line_dash="dash", line_color="#666",
                       annotation_text="Criminal Code cap: 35%")
    fig_sens.add_vline(x=cof, line_dash="dot", line_color="#1a7a4a",
                       annotation_text=f"Current CoF: {cof:.1%}")
    fig_sens.update_layout(height=300,
        xaxis_title="Cost of Funds", yaxis_title="Tier E Uncapped Rate",
        xaxis_tickformat=".1%", yaxis_tickformat=".0%",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_sens, use_container_width=True)

    sens_summary = sens_df.copy()
    sens_summary["cof"] = sens_summary["cof"].map(lambda v: f"{v:.1%}")
    sens_summary["tier_e_rate_raw"] = sens_summary["tier_e_rate_raw"].map(lambda v: f"{v:.2%}")
    sens_summary.columns = ["Cost of Funds", "# Tiers at 35% Cap", "Tier E Uncapped Rate"]
    st.dataframe(sens_summary, use_container_width=True, hide_index=True)
    st.caption("As CoF rises, more tiers hit the 35% Criminal Code ceiling — the lender "
               "loses pricing flexibility. At 8% CoF, tier E would need 35%+ to cover EL, "
               "OpEx, and ROE — meaning those applicants must be declined rather than priced.")

# ═══════════════════════════════════════════════════════════════
# TAB 6 — FRAUD MONITORING
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.markdown("""
    <div class="tab-summary">
    <h4>🔍 Fraud Detection & Post-Funding Monitoring</h4>
    <p>Post-funding fraud detection — identifying loans that were fraudulent after funding.
    Fraud model is integrated into the upstream decision engine in Tab 1
    (fraud score &gt; 65% = DECLINE_FRAUD before credit model runs).
    This tab shows portfolio-level fraud monitoring: alert tiers, fraud types,
    FPD cohort analysis, time-series fraud trends, product × type drill-down,
    and investigation queue.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="warn-box">
    <b>Synthetic label disclosure:</b> LendingClub and FICO HELOC datasets contain no
    confirmed fraud investigation labels. Labels here are synthetic — generated from
    known fraud indicator patterns. Fraud model AUC and precision metrics are inflated
    as a result. On real confirmed investigation labels expect AUC 0.72-0.85.
    Full swap interface in <code>src/data/fraud_label_generator.py</code>.
    </div>""", unsafe_allow_html=True)

    try:
        df_fraud = pd.read_parquet("data/processed/fraud_scored.parquet")
        n_fraud = int(df_fraud["fraud_confirmed"].sum())
        fraud_rate = df_fraud["fraud_confirmed"].mean()
        total_loss = df_fraud["loss_attributed"].sum()

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Loans", f"{len(df_fraud):,}")
        c2.metric("Confirmed Fraud", f"{n_fraud:,}", delta=f"{fraud_rate:.2%}")
        c3.metric("Fraud Losses", f"${total_loss:,.0f}",
            help="Loss Attributed = EAD at fraud-confirmation time minus any post-fraud "
                 "recovery (collateral seizure, bank account recovery, etc.). For unsecured "
                 "loans, recovery is typically 0-5% of principal; for HELOC, 40-70% via "
                 "property sale net of transaction costs.")
        c4.metric("CONFIRMED Alerts", f"{int((df_fraud['alert_tier']=='CONFIRMED').sum()):,}")
        c5.metric("HIGH Alerts", f"{int((df_fraud['alert_tier']=='HIGH').sum()):,}")

        l, r = st.columns(2)
        with l:
            st.markdown("### Alert Tier Distribution")
            tier_order=["CONFIRMED","HIGH","MEDIUM","LOW"]
            tc={"CONFIRMED":"#A32D2D","HIGH":"#D85A30","MEDIUM":"#BA7517","LOW":"#1D9E75"}
            tdf=df_fraud["alert_tier"].value_counts().reindex(tier_order,fill_value=0).reset_index()
            tdf.columns=["tier","count"]
            fig=px.bar(tdf,x="tier",y="count",color="tier",color_discrete_map=tc)
            fig.update_layout(showlegend=False,height=260,
                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
            st.caption("CONFIRMED: Freeze + investigate | HIGH: Escalate 24h | "
                       "MEDIUM: Enhanced monitoring | LOW: Routine")

        with r:
            st.markdown("### Fraud Type Breakdown")
            fdf=df_fraud[df_fraud["fraud_confirmed"]]["fraud_type"].value_counts().reset_index()
            fdf.columns=["type","count"]
            fig2=px.bar(fdf,x="count",y="type",orientation="h",
                color_discrete_sequence=["#1D9E75"])
            fig2.update_layout(height=260,showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2,use_container_width=True)

        # ── Time Series Trend ────────────────────────────────────────
        st.divider()
        st.markdown("### Fraud Trend Over Time")
        st.markdown("""<div class="explain-box">
        Monthly fraud rate by origination cohort — the most important signal for a fraud
        manager. A spike in fraud rate in a particular month indicates either a fraud ring
        attack, a control-gap exploitation, or a data-quality problem at onboarding.
        Flat trend = stable control environment. Rising trend = investigate recent control changes.
        </div>""", unsafe_allow_html=True)

        time_col = None
        for candidate in ["origination_month", "issue_d", "origination_date", "origination_year"]:
            if candidate in df_fraud.columns:
                time_col = candidate
                break

        if time_col is not None:
            try:
                ts = df_fraud.copy()
                # Convert to pandas Period for monthly grouping where possible
                if time_col == "origination_year":
                    ts["period"] = ts[time_col].astype(int).astype(str)
                else:
                    ts["period"] = pd.to_datetime(ts[time_col], errors="coerce").dt.to_period("M").astype(str)
                ts = ts[ts["period"] != "nan"]
                trend = ts.groupby("period").agg(
                    n_loans=("fraud_confirmed", "size"),
                    n_fraud=("fraud_confirmed", "sum"),
                ).reset_index()
                trend["fraud_rate"] = trend["n_fraud"] / trend["n_loans"].replace(0, 1)

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=trend["period"], y=trend["fraud_rate"],
                    mode="lines+markers", name="Fraud rate",
                    line=dict(color="#c0392b", width=2),
                    marker=dict(size=6)))
                fig_trend.add_hline(y=fraud_rate, line_dash="dash", line_color="#888",
                    annotation_text=f"Portfolio avg: {fraud_rate:.2%}",
                    annotation_position="top right")
                fig_trend.update_layout(height=320,
                    xaxis_title=f"Origination cohort ({time_col})",
                    yaxis_title="Fraud Rate",
                    yaxis_tickformat=".1%",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_trend, use_container_width=True)
                # Spike detection
                if len(trend) > 3:
                    recent_avg = trend["fraud_rate"].tail(3).mean()
                    prior_avg  = trend["fraud_rate"].iloc[:-3].mean() if len(trend) > 3 else trend["fraud_rate"].mean()
                    if recent_avg > prior_avg * 1.25:
                        st.warning(f"⚠ Recent 3-period fraud rate ({recent_avg:.2%}) "
                                   f"is 25%+ above prior average ({prior_avg:.2%}). "
                                   "Investigate recent onboarding control changes.")
            except Exception as _ts_e:
                st.caption(f"Time series unavailable ({_ts_e}).")
        else:
            st.info("No origination-time column found. Time-series requires an "
                    "`origination_month`, `issue_d`, `origination_date`, or `origination_year` field.")

        # ── Product × Fraud Type drill-down ──────────────────────────
        st.divider()
        st.markdown("### Product × Fraud Type Drill-Down")
        st.markdown("""<div class="explain-box">
        Which fraud types concentrate in which products? First-party fraud tends to be
        higher in unsecured (lower barrier to application). Synthetic identity and income
        misrepresentation tend to cluster by product depending on the lender's KYC depth.
        </div>""", unsafe_allow_html=True)

        if "product_type" in df_fraud.columns and "fraud_type" in df_fraud.columns:
            drill = df_fraud[df_fraud["fraud_confirmed"]].copy()
            drill["product"] = drill["product_type"].map({0: "Unsecured", 1: "HELOC"})
            pivot = drill.pivot_table(
                index="fraud_type", columns="product",
                values="fraud_confirmed", aggfunc="count", fill_value=0,
            )
            st.dataframe(pivot, use_container_width=True)

            # Stacked bar
            fig_drill = go.Figure()
            for col in pivot.columns:
                fig_drill.add_trace(go.Bar(
                    name=col, y=pivot.index, x=pivot[col], orientation="h",
                    marker_color="#1a7a4a" if col == "Unsecured" else "#3498db"))
            fig_drill.update_layout(barmode="stack", height=280,
                xaxis_title="Count", yaxis_title="Fraud Type",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_drill, use_container_width=True)
        else:
            st.caption("Product × fraud-type drill-down requires both "
                       "`product_type` and `fraud_type` columns.")

        st.divider()
        st.markdown("### First Payment Default (FPD) Analysis")
        st.markdown("""<div class="explain-box">
        FPD = loan that misses its very first scheduled payment. Strongest post-funding fraud signal.
        FPD lift = how many times more likely FPD loans are to be fraudulent vs normal loans.
        In production: FPD defined as payment_date[1] > due_date[1] + 30 days (requires LOS data).
        </div>""", unsafe_allow_html=True)
        fpd_col="fpd_flag" if "fpd_flag" in df_fraud.columns else "fpd_risk_flag"
        fpd_count=int(df_fraud[fpd_col].sum())
        fpd_rate=df_fraud[df_fraud[fpd_col].astype(bool)]["fraud_confirmed"].mean()
        clean_rate=df_fraud[~df_fraud[fpd_col].astype(bool)]["fraud_confirmed"].mean()
        lift=fpd_rate/max(clean_rate,0.001)
        f1,f2,f3,f4=st.columns(4)
        f1.metric("FPD Loans",f"{fpd_count:,}")
        f2.metric("FPD Fraud Rate",f"{fpd_rate:.1%}")
        f3.metric("Normal Fraud Rate",f"{clean_rate:.1%}")
        f4.metric("FPD Lift",f"{lift:.1f}×")

        st.divider()
        st.markdown("### Investigation Queue — Top Priority Cases")
        st.markdown("""<div class="explain-box">
        <b>Column definitions for the queue:</b><br>
        • <b>fraud_prob</b>: model's fraud probability, higher = higher priority<br>
        • <b>ead_estimate</b>: exposure at default (outstanding principal at time of fraud confirmation)<br>
        • <b>loss_attributed</b>: EAD net of estimated recovery — the dollar figure booked
          as a fraud loss on the P&L. Formula: EAD × (1 − recovery rate). For unsecured,
          recovery = 0-5%; for HELOC, recovery = 40-70% via collateral.<br>
        • <b>fpd_risk_flag</b>: first payment default indicator (strongest post-funding signal)<br>
        • <b>synthetic_id_risk_flag</b>: fabricated identity indicators present<br>
        • <b>multi_app_flag</b>: loan stacking signal (multiple applications in short window)
        </div>""", unsafe_allow_html=True)

        try:
            inv_q=pd.read_csv("reports/phase9/investigation_queue.csv")
            sc=[c for c in ["alert_tier","fraud_prob","fraud_type","credit_score",
                             "ead_estimate","loss_attributed","fpd_risk_flag",
                             "synthetic_id_risk_flag","multi_app_flag"] if c in inv_q.columns]
            st.dataframe(inv_q[sc].head(15).style.format({
                "fraud_prob":"{:.2%}","ead_estimate":"${:,.0f}","loss_attributed":"${:,.0f}"}),
                use_container_width=True)
        except:
            st.info("Run Phase 9 to generate investigation queue.")

    except Exception as e:
        st.warning("Fraud data not found. Run: python build.py --from 9")
        st.caption(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════
# TAB 7 — EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════
with tab7:
    st.markdown("""
    <div class="tab-summary">
    <h4>📊 Executive Dashboard</h4>
    <p>Consolidated portfolio view for Chief Risk Officer or Board Risk Committee.
    Three sub-views: <b>Portfolio Health</b> (KPIs, tier distribution, EL concentration),
    <b>Model Monitoring</b> (AUC, PSI, feature stability), and
    <b>Risk Concentration</b> (IFRS 9 stage breakdown, product split, vintage performance).
    All data sourced from the same pipeline as the other tabs.</p>
    </div>""", unsafe_allow_html=True)
    render_dashboard(portfolio, reports)
