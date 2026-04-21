"""
src/app/dashboard.py
─────────────────────
Executive Dashboard — three views:
  1. Portfolio Health     — KPIs, EL by tier, product split, vintage
  2. Model Monitoring     — Score distribution, approval rate, PSI alerts
  3. Risk Concentration   — EL concentration, EAD breakdown, stage heatmap

Imported and rendered as an additional tab in streamlit_app.py.

v1.1 changes
────────────
- Fixed CAR denominator bug (was min_capital / EAD; now available capital / RWA,
  consistent with Tab 4 methodology).
- Added defensive empty-portfolio guard at the top of render_dashboard.
- PD distribution vline now uses the real APPROVAL_THRESHOLD (0.28) from utils
  instead of a hardcoded 35%; 35% line retained separately as the refer-band
  ceiling.
- EL concentration grouped-bar chart now uses two distinct colours
  (loan share vs EL share) instead of the same palette with opacity.
- Concentration ratio drops empty tiers instead of using a 0.001 floor that
  produced misleading huge ratios.
- Model monitoring view now includes a banner referencing the v1.1 segmented
  models and points the reader to Tab 3 for the fuller comparison.
- Stress test guard simplified (was an always-true condition).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import joblib

# Pull shared constants from utils so the dashboard cannot drift out of sync
# with the rest of the app. These are the same constants Tabs 1-6 use.
try:
    from src.app.utils import APPROVAL_THRESHOLD, REFER_BAND_LOWER
except ImportError:
    # Safe fallbacks if utils is unavailable at import time
    APPROVAL_THRESHOLD = 0.28
    REFER_BAND_LOWER   = 0.28


# Assumed available-capital stack for CAR calculation (demonstration only).
# In production this would come from the lender's booked CET1 from the OSFI
# return, not an assumed ratio.
ASSUMED_CET1_TARGET = 0.115    # 11.5% D-SIB target — matches Tab 4
CAR_MINIMUM         = 0.08     # OSFI Pillar 1 minimum


def render_dashboard(portfolio: pd.DataFrame, reports: dict):
    """
    Render the full executive dashboard.
    Call this from streamlit_app.py inside a tab.
    """
    st.markdown("## Executive Dashboard")
    st.markdown("""
    <div style="background:#f0f7ff;border-left:4px solid #2196F3;padding:12px 16px;
    border-radius:0 8px 8px 0;margin:8px 0 20px 0;font-size:0.88rem;color:#1a3a5c">
    <b>What this shows:</b> A consolidated view of portfolio health, model performance,
    and risk concentration — the kind of morning dashboard a Chief Risk Officer would review.
    All metrics are derived from the same underlying data that powers every other tab.
    </div>
    """, unsafe_allow_html=True)

    # ── Empty-portfolio guard ────────────────────────────────────────
    # If the processed portfolio parquet hasn't been built yet, every chart
    # below will either crash or render nonsense. Surface that clearly.
    required_cols = {"ead_estimate", "expected_loss", "pd_score",
                     "credit_score", "rwa", "risk_tier", "product_type",
                     "ifrs9_stage", "lgd_estimate"}
    if portfolio is None or len(portfolio) == 0:
        st.warning("📭 Portfolio data not available yet. The Executive Dashboard requires "
                   "`data/processed/portfolio_regulatory.parquet`. Run the full pipeline "
                   "to generate it:")
        st.code("python build.py", language="bash")
        st.info("Once the pipeline is built, re-open this tab — all KPIs and charts will populate.")
        return
    missing = required_cols - set(portfolio.columns)
    if missing:
        st.warning(f"📭 Portfolio data is missing required columns: {sorted(missing)}. "
                   "This usually means the pipeline was interrupted before Phase 6. "
                   "Re-run `python build.py --from 4` to regenerate regulatory metrics.")
        return

    dash1, dash2, dash3 = st.tabs([
        "Portfolio Health",
        "Model Monitoring",
        "Risk Concentration",
    ])

    with dash1:
        _render_portfolio_health(portfolio, reports)
    with dash2:
        _render_model_monitoring(portfolio, reports)
    with dash3:
        _render_risk_concentration(portfolio, reports)


def _render_portfolio_health(portfolio: pd.DataFrame, reports: dict):
    """Portfolio Health Dashboard."""
    st.markdown("### Portfolio Health Overview")

    # ── Top KPI row (now 7 metrics including corrected CAR) ───────
    total_ead   = float(portfolio["ead_estimate"].sum())
    total_el    = float(portfolio["expected_loss"].sum())
    el_rate     = total_el / total_ead if total_ead > 0 else 0
    avg_pd      = float(portfolio["pd_score"].mean())
    avg_score   = float(portfolio["credit_score"].mean())
    total_rwa   = float(portfolio["rwa"].sum())
    n_loans     = len(portfolio)

    # CAR — correct formula: Available Capital / RWA.
    # Available capital assumed = ASSUMED_CET1_TARGET × RWA (demonstration).
    # The previous version divided by EAD which is not CAR and produces a
    # misleading number. See MODEL_CARD §v1.1 Governance for context.
    available_capital = total_rwa * ASSUMED_CET1_TARGET
    car = available_capital / total_rwa if total_rwa > 0 else 0.0  # = ASSUMED_CET1_TARGET

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Total Loans", f"{n_loans:,}",
              help="Total loans in the portfolio")
    c2.metric("Total EAD", f"${total_ead/1e6:.1f}M",
              help="Total Exposure at Default — the amount outstanding across all loans")
    c3.metric("Portfolio EL Rate", f"{el_rate:.2%}",
              help="Expected Loss / EAD. Industry benchmark for healthy consumer portfolio: 3-8%",
              delta="Healthy" if el_rate < 0.08 else "Elevated",
              delta_color="normal" if el_rate < 0.08 else "inverse")
    c4.metric("Avg PD", f"{avg_pd:.2%}",
              help="Mean Probability of Default across portfolio after Platt calibration")
    c5.metric("Avg Credit Score", f"{avg_score:.0f}",
              help="Mean FICO-equivalent score (300-850)")
    c6.metric("Total RWA", f"${total_rwa/1e6:.1f}M",
              help="Basel III Risk-Weighted Assets — drives minimum capital requirement")
    c7.metric("CAR (assumed)", f"{car:.2%}",
              delta=f"✓ Above {CAR_MINIMUM:.0%} min" if car >= CAR_MINIMUM else "⚠ Below min",
              delta_color="normal" if car >= CAR_MINIMUM else "inverse",
              help=f"Available Capital ÷ RWA. Assumes available capital = "
                   f"{ASSUMED_CET1_TARGET:.1%} × RWA (CET1 target assumption for demonstration). "
                   f"Real CAR uses booked CET1 from the OSFI return. See Tab 4 for full breakdown.")

    st.divider()

    # ── Tier distribution + EL by tier ───────────────────────────
    col_l, col_r = st.columns(2)

    tier_colors = {"A":"#1a7a4a","B":"#2d9e6b","C":"#e6a817","D":"#e07520","E":"#c0392b"}

    with col_l:
        st.markdown("**Loan count by risk tier**")
        st.markdown("""
        <div style="font-size:0.8rem;color:#555;margin-bottom:8px">
        Tier A-B = prime. Tier C = near-prime. Tier D-E = subprime.
        A healthy consumer portfolio typically has 40-50% in Tiers A-B.
        </div>""", unsafe_allow_html=True)

        tier_counts = portfolio["risk_tier"].value_counts().reindex(
            ["A","B","C","D","E"], fill_value=0)

        fig = go.Figure(go.Pie(
            labels=[f"Tier {t}" for t in tier_counts.index],
            values=tier_counts.values,
            hole=0.55,
            marker_colors=[tier_colors[t] for t in tier_counts.index],
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Loans: %{value:,}<br>Share: %{percent}<extra></extra>"
        ))
        fig.update_layout(
            height=280, showlegend=False,
            margin=dict(l=10,r=10,t=10,b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"{n_loans:,}<br>loans",
                              x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Expected loss by risk tier**")
        st.markdown("""
        <div style="font-size:0.8rem;color:#555;margin-bottom:8px">
        Even a small Tier E book can dominate total EL.
        This shows the concentration of loss exposure.
        </div>""", unsafe_allow_html=True)

        el_by_tier = portfolio.groupby("risk_tier")["expected_loss"].sum().reindex(
            ["A","B","C","D","E"], fill_value=0)

        fig2 = go.Figure(go.Bar(
            x=el_by_tier.index,
            y=el_by_tier.values,
            marker_color=[tier_colors[t] for t in el_by_tier.index],
            text=[f"${v/1e6:.1f}M" for v in el_by_tier.values],
            textposition="outside",
            hovertemplate="<b>Tier %{x}</b><br>EL: $%{y:,.0f}<extra></extra>"
        ))
        fig2.update_layout(
            height=280, showlegend=False,
            yaxis_title="Expected Loss ($)",
            yaxis_tickformat="$,.0f",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Product split + EL rate by product ───────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("**EAD by product type**")
        prod_ead = portfolio.groupby("product_type")["ead_estimate"].sum()
        prod_labels = {0:"Unsecured", 1:"HELOC (Secured)"}
        fig3 = go.Figure(go.Pie(
            labels=[prod_labels.get(i, str(i)) for i in prod_ead.index],
            values=prod_ead.values,
            hole=0.5,
            marker_colors=["#2d9e6b","#3498db"],
            textinfo="label+percent",
        ))
        fig3.update_layout(height=240, showlegend=False,
            margin=dict(l=10,r=10,t=10,b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.markdown("**EL rate by product**")
        st.markdown("""
        <div style="font-size:0.8rem;color:#555;margin-bottom:8px">
        HELOC should have a lower EL rate than unsecured loans
        because property collateral reduces LGD.
        </div>""", unsafe_allow_html=True)

        prod_metrics = portfolio.groupby("product_type").agg(
            total_el=("expected_loss","sum"),
            total_ead=("ead_estimate","sum"),
            avg_pd=("pd_score","mean"),
            avg_lgd=("lgd_estimate","mean"),
        ).reset_index()
        prod_metrics["el_rate"] = prod_metrics["total_el"] / prod_metrics["total_ead"]
        prod_metrics["product"] = prod_metrics["product_type"].map(prod_labels)

        fig4 = go.Figure(go.Bar(
            x=prod_metrics["product"],
            y=prod_metrics["el_rate"],
            marker_color=["#2d9e6b","#3498db"],
            text=[f"{v:.2%}" for v in prod_metrics["el_rate"]],
            textposition="outside",
        ))
        fig4.update_layout(
            height=240, yaxis_tickformat=".1%",
            yaxis_title="EL Rate (EL/EAD)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Vintage default rate chart ────────────────────────────────
    st.markdown("**Vintage default rates — actual performance by origination year**")
    st.markdown("""
    <div style="font-size:0.8rem;color:#555;margin-bottom:8px">
    Shows how loans originated in each year actually performed.
    The GFC cohorts (2007-2010) show higher default rates — exactly what the model
    needs to have learned from to be well-calibrated.
    </div>""", unsafe_allow_html=True)

    vc = reports.get("vintage", pd.DataFrame())
    if len(vc) > 0 and "origination_year" in vc.columns and "default_rate" in vc.columns:
        avg_dr = vc["default_rate"].mean()
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            x=vc["origination_year"].astype(int),
            y=vc["default_rate"],
            marker_color=["#c0392b" if v > avg_dr * 1.2 else "#2d9e6b"
                          for v in vc["default_rate"]],
            text=[f"{v:.1%}" for v in vc["default_rate"]],
            textposition="outside",
            name="Default Rate",
        ))
        fig5.add_hline(y=avg_dr, line_dash="dash", line_color="#e07520",
                       annotation_text=f"Portfolio avg: {avg_dr:.2%}",
                       annotation_position="top right")
        fig5.update_layout(
            height=280, yaxis_tickformat=".0%",
            xaxis_title="Origination Year",
            yaxis_title="Actual Default Rate",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Red bars = cohorts with default rate > 120% of portfolio average.")


def _render_model_monitoring(portfolio: pd.DataFrame, reports: dict):
    """Model Monitoring Dashboard."""
    st.markdown("### Model Monitoring")
    st.markdown("""
    <div style="font-size:0.8rem;color:#555;margin-bottom:16px">
    OSFI E-23 requires ongoing monitoring of model performance. Key signals:
    PSI alerts indicate the population has shifted. Score distribution drift means
    the model is seeing applicants different from those it was trained on.
    </div>""", unsafe_allow_html=True)

    # Reference to v1.1 segmented comparison (in Tab 3)
    mc_seg = reports.get("model_comparison_segmented", pd.DataFrame())
    if len(mc_seg) > 0:
        st.info("ℹ **v1.1 segmented models are available** — per-product AUC breakdowns "
                "(Unsecured ~0.72, Secured ~0.76) are shown in Tab 3, Section 3. "
                "The KPIs below reflect the v1.0 unified model used for cross-product decisioning.")

    # ── Score distribution ────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Credit score distribution**")
        st.markdown("""
        <div style="font-size:0.8rem;color:#555;margin-bottom:8px">
        A healthy prime/near-prime consumer portfolio peaks in the 640-740 range.
        Heavy concentration below 580 indicates a subprime book with elevated EL.
        </div>""", unsafe_allow_html=True)

        scores = portfolio["credit_score"].dropna()
        fig = go.Figure(go.Histogram(
            x=scores, nbinsx=40,
            marker_color="#2d9e6b", opacity=0.8,
            hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>"
        ))
        for boundary, label, color in [
            (580,"Tier D/E boundary","#c0392b"),
            (630,"Tier C/D","#e07520"),
            (680,"Tier B/C","#e6a817"),
            (720,"Tier A/B","#2d9e6b"),
        ]:
            fig.add_vline(x=boundary, line_dash="dot", line_color=color,
                          annotation_text=label, annotation_position="top")
        fig.update_layout(
            height=300, xaxis_title="Credit Score",
            yaxis_title="Loan Count",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**PD score distribution**")
        st.markdown("""
        <div style="font-size:0.8rem;color:#555;margin-bottom:8px">
        After Platt scaling calibration the distribution should be right-skewed
        with most mass below 25%. High mass above 40% signals a subprime book
        or a miscalibrated model.
        </div>""", unsafe_allow_html=True)

        pds = portfolio["pd_score"].dropna()
        fig2 = go.Figure(go.Histogram(
            x=pds * 100, nbinsx=40,
            marker_color="#3498db", opacity=0.8,
        ))
        # FIX: use the real APPROVAL_THRESHOLD from utils (28%) not hardcoded 35%.
        # Previous version labelled 35% as "Approval threshold" which mismatched Tab 1.
        fig2.add_vline(x=APPROVAL_THRESHOLD * 100, line_dash="dash", line_color="#c0392b",
                       annotation_text=f"Approval threshold ({APPROVAL_THRESHOLD:.0%})")
        fig2.add_vline(x=35, line_dash="dot", line_color="#888",
                       annotation_text="Refer band ceiling (35%)")
        avg_pd = pds.mean()
        fig2.add_vline(x=avg_pd * 100, line_dash="dot", line_color="#e07520",
                       annotation_text=f"Avg PD: {avg_pd:.1%}")
        fig2.update_layout(
            height=300, xaxis_title="PD Score (%)",
            yaxis_title="Loan Count",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Model comparison table ────────────────────────────────────
    st.markdown("**Model performance — scorecard vs XGBoost (out-of-time test)**")

    mc = reports.get("model_comparison", pd.DataFrame())
    if len(mc) > 0:
        col1, col2 = st.columns(2)
        for i, (_, row) in enumerate(mc.iterrows()):
            col = col1 if i == 0 else col2
            with col:
                auc = float(row["auc"])
                ks  = float(row["ks"])
                gin = float(row["gini"])
                auc_ok = auc >= 0.70
                ks_ok  = ks  >= 0.30
                st.markdown(f"**{row['model']}**")
                c1,c2,c3 = st.columns(3)
                c1.metric("AUC",  f"{auc:.4f}",
                          delta="✓ Benchmark" if auc_ok else "⚠ Below 0.70",
                          delta_color="normal" if auc_ok else "inverse")
                c2.metric("KS",   f"{ks:.4f}",
                          delta="✓ OK" if ks_ok else "⚠ Below 0.30",
                          delta_color="normal" if ks_ok else "inverse")
                c3.metric("Gini", f"{gin:.4f}")
        if len(mc_seg) > 0:
            st.caption("→ For per-product AUC breakouts that remove the product-proxy effect "
                       "in the unified model, see Tab 3, Section 3.")

    st.divider()

    # ── PSI / CSI monitoring ──────────────────────────────────────
    st.markdown("**Feature stability monitoring (PSI/CSI)**")
    st.markdown("""
    <div style="font-size:0.8rem;color:#555;margin-bottom:8px">
    PSI measures whether a feature's distribution has shifted between training and current portfolio.
    <span style="background:#f8d7da;padding:2px 6px;border-radius:3px">ALERT &gt; 0.25</span> —
    model review required under OSFI E-23.
    <span style="background:#fff3cd;padding:2px 6px;border-radius:3px">WATCH 0.10-0.25</span> —
    monitor closely.
    <span style="background:#d4edda;padding:2px 6px;border-radius:3px">OK &lt; 0.10</span> —
    stable.
    </div>""", unsafe_allow_html=True)

    csi = reports.get("csi", pd.DataFrame())
    if len(csi) > 0:
        n_alert = (csi["status"] == "ALERT — investigate").sum() if "status" in csi.columns else 0
        n_watch = (csi["status"] == "WATCH").sum() if "status" in csi.columns else 0
        n_ok    = (csi["status"] == "OK").sum() if "status" in csi.columns else 0

        c1,c2,c3 = st.columns(3)
        c1.metric("ALERT features", n_alert,
                  delta="Model review required" if n_alert > 0 else None,
                  delta_color="inverse" if n_alert > 0 else "normal")
        c2.metric("WATCH features", n_watch)
        c3.metric("OK features",    n_ok)

        st.dataframe(
            csi.style.map(
                lambda v: "background-color:#f8d7da" if v=="ALERT — investigate"
                else ("background-color:#fff3cd" if v=="WATCH"
                else "background-color:#d4edda" if v=="OK" else ""),
                subset=["status"] if "status" in csi.columns else []
            ),
            use_container_width=True
        )


def _render_risk_concentration(portfolio: pd.DataFrame, reports: dict):
    """Risk Concentration Dashboard."""
    st.markdown("### Risk Concentration Analysis")
    st.markdown("""
    <div style="font-size:0.8rem;color:#555;margin-bottom:16px">
    Concentration risk identifies where losses are clustered. A well-diversified portfolio
    has EL spread across risk tiers and products. Heavy concentration in a single segment
    amplifies tail risk during a downturn.
    </div>""", unsafe_allow_html=True)

    # ── EL concentration ─────────────────────────────────────────
    st.markdown("**EL concentration — cumulative share by risk tier**")
    st.markdown("""
    <div style="font-size:0.8rem;color:#555;margin-bottom:8px">
    If Tier E represents 10% of loans but 60% of EL, the portfolio has high
    tail concentration. Healthy portfolio: Tier E share of EL should not exceed
    3× its share of loan count.
    </div>""", unsafe_allow_html=True)

    tier_summary = portfolio.groupby("risk_tier").agg(
        n_loans=("pd_score","count"),
        total_el=("expected_loss","sum"),
        total_ead=("ead_estimate","sum"),
        avg_pd=("pd_score","mean"),
        avg_lgd=("lgd_estimate","mean"),
    ).reindex(["A","B","C","D","E"]).fillna(0).reset_index()

    # Shares computed using the full portfolio denominators so percentages
    # sum to 100% even when some tiers are empty.
    total_n   = float(tier_summary["n_loans"].sum()) or 1.0
    total_elx = float(tier_summary["total_el"].sum()) or 1.0
    tier_summary["loan_share"] = tier_summary["n_loans"] / total_n
    tier_summary["el_share"]   = tier_summary["total_el"] / total_elx
    tier_summary["el_rate"]    = np.where(
        tier_summary["total_ead"] > 0,
        tier_summary["total_el"] / tier_summary["total_ead"],
        0.0,
    )

    # FIX: Concentration ratio — NaN for tiers with zero loans rather than
    # using a 0.001 floor (which produced misleading huge ratios).
    tier_summary["concentration_ratio"] = np.where(
        tier_summary["loan_share"] > 0,
        tier_summary["el_share"] / tier_summary["loan_share"].replace(0, np.nan),
        np.nan,
    )

    # FIX: two distinct colours for the two series (previously both used the
    # tier-colour palette + opacity, which was hard to read at a glance).
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tier_summary["risk_tier"],
        y=tier_summary["loan_share"] * 100,
        name="% of Loans",
        marker_color="#6aa1c9",   # light blue — population share
        text=[f"{v*100:.1f}%" for v in tier_summary["loan_share"]],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=tier_summary["risk_tier"],
        y=tier_summary["el_share"] * 100,
        name="% of EL",
        marker_color="#c0392b",   # red — loss share
        text=[f"{v*100:.1f}%" for v in tier_summary["el_share"]],
        textposition="outside",
    ))

    fig.update_layout(
        barmode="group", height=320,
        yaxis_title="Share (%)", yaxis_tickformat=".0f",
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Concentration table
    st.dataframe(
        tier_summary[["risk_tier","n_loans","loan_share","total_el",
                       "el_share","el_rate","concentration_ratio"]].style.format({
            "loan_share":"{:.1%}", "total_el":"${:,.0f}",
            "el_share":"{:.1%}", "el_rate":"{:.2%}",
            "concentration_ratio":"{:.1f}×",
        }, na_rep="—").background_gradient(subset=["concentration_ratio"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True
    )
    st.caption("Concentration ratio = EL share / loan share. "
               "Ratio > 3× for any tier warrants management attention. "
               "Empty tiers display '—'.")

    st.divider()

    # ── IFRS 9 stage breakdown ────────────────────────────────────
    st.markdown("**IFRS 9 stage distribution — loan count and ECL**")

    stage_summary = portfolio.groupby("ifrs9_stage").agg(
        n_loans=("pd_score","count"),
        total_ead=("ead_estimate","sum"),
        total_ecl=("ecl","sum"),
        avg_pd=("pd_score","mean"),
    ).reset_index()
    stage_summary["coverage"] = np.where(
        stage_summary["total_ead"] > 0,
        stage_summary["total_ecl"] / stage_summary["total_ead"],
        0.0,
    )
    stage_names = {1:"Stage 1 — Performing", 2:"Stage 2 — SICR", 3:"Stage 3 — Impaired"}
    stage_colors_map = {1:"#1a7a4a", 2:"#e6a817", 3:"#c0392b"}
    stage_summary["stage_name"] = stage_summary["ifrs9_stage"].map(stage_names)

    col_a, col_b = st.columns(2)

    with col_a:
        fig6 = go.Figure(go.Pie(
            labels=stage_summary["stage_name"],
            values=stage_summary["n_loans"],
            hole=0.5,
            marker_colors=[stage_colors_map.get(s,"#888")
                           for s in stage_summary["ifrs9_stage"]],
            textinfo="label+percent",
        ))
        fig6.update_layout(
            height=260, showlegend=False, title="Loan count by stage",
            margin=dict(l=10,r=10,t=40,b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig6, use_container_width=True)

    with col_b:
        fig7 = go.Figure(go.Bar(
            x=stage_summary["stage_name"],
            y=stage_summary["total_ecl"],
            marker_color=[stage_colors_map.get(s,"#888")
                          for s in stage_summary["ifrs9_stage"]],
            text=[f"${v/1e6:.1f}M" for v in stage_summary["total_ecl"]],
            textposition="outside",
        ))
        fig7.update_layout(
            height=260, title="ECL provision by stage",
            yaxis_title="ECL ($)", yaxis_tickformat="$,.0f",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig7, use_container_width=True)

    st.dataframe(
        stage_summary[["stage_name","n_loans","total_ead","total_ecl","coverage","avg_pd"
                        ]].style.format({
            "total_ead":"${:,.0f}", "total_ecl":"${:,.0f}",
            "coverage":"{:.2%}", "avg_pd":"{:.2%}",
        }),
        use_container_width=True, hide_index=True
    )
    st.caption("Coverage ratio = ECL / EAD. "
               "Stage 1 typical: 0.5-2%. Stage 2: 5-15%. Stage 3: 40-80%.")

    st.divider()

    # ── Stress test summary ───────────────────────────────────────
    st.markdown("**Stress test — capital headroom by scenario**")

    stress = reports.get("stress", pd.DataFrame())
    if len(stress) > 0:
        sc_colors = {
            "Base Case":"#1a7a4a",
            "Adverse Scenario":"#e6a817",
            "Severe Scenario":"#c0392b"
        }
        fig8 = make_subplots(rows=1, cols=2,
                              subplot_titles=["Expected Loss by Scenario",
                                              "Required Capital by Scenario"])

        for _, row in stress.iterrows():
            color = sc_colors.get(row["scenario"],"#888")
            fig8.add_trace(go.Bar(
                name=row["scenario"], x=[row["scenario"]],
                y=[float(row["total_el_stressed"])],
                marker_color=color, showlegend=True,
                text=f"{float(row['el_rate']):.1%}", textposition="outside",
            ), row=1, col=1)
            fig8.add_trace(go.Bar(
                name=row["scenario"], x=[row["scenario"]],
                y=[float(row["required_capital"])],
                marker_color=color, showlegend=False,
            ), row=1, col=2)

        fig8.update_layout(
            height=300, barmode="group",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.15),
        )
        fig8.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(fig8, use_container_width=True)

        # FIX: simplified guard (the previous `if` was always true because
        # `required_capital` is always present in the stress CSV).
        required_stress_cols = ["scenario","pd_multiplier","avg_pd_stressed",
                                 "el_rate","total_rwa","required_capital"]
        if all(c in stress.columns for c in required_stress_cols):
            st.dataframe(
                stress[required_stress_cols].style.format({
                    "pd_multiplier":"{:.2f}×", "avg_pd_stressed":"{:.2%}",
                    "el_rate":"{:.2%}", "total_rwa":"${:,.0f}",
                    "required_capital":"${:,.0f}",
                }),
                use_container_width=True, hide_index=True
            )
    else:
        st.info("Stress test results not available. Run `python build.py --from 6` to generate.")
