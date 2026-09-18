"""
Executive Summary — US Healthcare C-Suite Dashboard
CMS HRRP alignment, financial impact, quality metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── US National Benchmarks (CMS / AHRQ 2023) ──────────────────────────────────
NATIONAL_HF_READMIT   = 22.0   # %
CMS_PENALTY_THRESHOLD =  1.0   # excess readmission ratio threshold
AVG_READMIT_COST      = 15000  # USD per readmission
HRRP_MAX_PENALTY      =  0.03  # 3% of base DRG payments


def show(full_data, train_data, test_data):
    st.markdown("## 🏠 Executive Summary — Hospital Leadership View")

    st.markdown("""
    <div class="story-banner">
    <strong>Why this matters:</strong> The Centers for Medicare & Medicaid Services (CMS) Hospital Readmissions
    Reduction Program (HRRP) penalizes hospitals up to <strong>3%</strong> of all Medicare DRG payments
    when 30-day readmission rates for Heart Failure, AMI, and Pneumonia exceed national benchmarks.
    For a 300-bed community hospital, this translates to <strong>$1M–$3M in annual penalties</strong>.
    This dashboard helps clinical and operational leaders identify high-risk patients before discharge and
    deploy targeted interventions — reducing readmissions, improving outcomes, and protecting reimbursement.
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    readmit_rate   = full_data['readmitted_30d'].mean() * 100
    total_patients = len(full_data)
    total_readmits = int(full_data['readmitted_30d'].sum())
    avg_los        = full_data['length_of_stay'].mean()
    avg_age        = full_data['age'].mean()
    avoided_readmits = max(0, int((NATIONAL_HF_READMIT - readmit_rate) / 100 * total_patients))
    est_savings    = avoided_readmits * AVG_READMIT_COST

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Patients",        f"{total_patients:,}")
    c2.metric("30-Day Readmit Rate",   f"{readmit_rate:.1f}%",
              delta=f"{readmit_rate - NATIONAL_HF_READMIT:+.1f}% vs national",
              delta_color="inverse")
    c3.metric("Total Readmissions",    f"{total_readmits:,}")
    c4.metric("Avg Length of Stay",    f"{avg_los:.1f} days")
    c5.metric("Avg Patient Age",       f"{avg_age:.0f} yrs")
    c6.metric("Est. Readmits Avoided", f"{avoided_readmits:,}",
              help="Patients not readmitted vs. national 22% benchmark")

    st.markdown("---")

    # ── Story: Performance vs National ───────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-title">📊 Performance vs. National Benchmarks</div>', unsafe_allow_html=True)

        categories = ["Your Hospital", "National Average (CMS)", "Top Quartile", "Top Decile"]
        rates      = [readmit_rate, NATIONAL_HF_READMIT, 17.5, 14.2]
        colors     = ["#1a6eb5", "#f39c12", "#27ae60", "#2ecc71"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories, y=rates,
            marker_color=colors,
            text=[f"{r:.1f}%" for r in rates],
            textposition="outside",
            width=0.5
        ))
        fig.add_hline(y=NATIONAL_HF_READMIT, line_dash="dash",
                      line_color="#e74c3c", annotation_text="CMS National Average")
        fig.update_layout(
            title="30-Day HF Readmission Rate — Benchmark Comparison",
            yaxis_title="Readmission Rate (%)", yaxis_range=[0, 30],
            height=380, plot_bgcolor="white",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#f0f0f0")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">💰 Financial Impact Estimator</div>', unsafe_allow_html=True)

        medicare_base = st.number_input("Annual Medicare DRG Revenue ($M)", value=45.0, step=1.0, min_value=1.0)
        medicare_base_usd = medicare_base * 1_000_000

        excess_ratio = max(0, readmit_rate / NATIONAL_HF_READMIT - 1)
        hrrp_penalty_pct = min(excess_ratio, HRRP_MAX_PENALTY)
        hrrp_penalty_usd = hrrp_penalty_pct * medicare_base_usd

        st.markdown(f"""
        <div class="alert-critical">
        <strong>⚠️ Estimated HRRP Penalty</strong><br>
        Excess Readmission Ratio: <strong>{1 + excess_ratio:.3f}</strong><br>
        Penalty Rate: <strong>{hrrp_penalty_pct*100:.2f}%</strong><br>
        Estimated Annual Penalty: <strong>${hrrp_penalty_usd:,.0f}</strong>
        </div>
        """, unsafe_allow_html=True)

        cost_per_readmit = st.number_input("Cost per Readmission ($)", value=15000, step=1000)
        total_readmit_cost = total_readmits * cost_per_readmit

        st.markdown(f"""
        <div class="alert-warning">
        <strong>📋 Total Readmission Cost Burden</strong><br>
        {total_readmits} readmissions × ${cost_per_readmit:,}<br>
        = <strong>${total_readmit_cost:,.0f}</strong>
        </div>
        """, unsafe_allow_html=True)

        reduction_target = st.slider("Readmission Reduction Target (%)", 5, 50, 20)
        savings = int(total_readmits * reduction_target / 100) * cost_per_readmit
        st.markdown(f"""
        <div class="alert-info">
        <strong>✅ Potential Annual Savings</strong><br>
        {reduction_target}% reduction = {int(total_readmits * reduction_target/100)} fewer readmissions<br>
        = <strong>${savings:,.0f} saved</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Quality Scorecard ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏆 Quality Scorecard — HRRP & CMS Measures</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Gauge: Readmission Rate
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=readmit_rate,
            title={"text": "HF 30-Day Readmit Rate"},
            delta={"reference": NATIONAL_HF_READMIT, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 35]},
                "bar": {"color": "#1a6eb5"},
                "steps": [
                    {"range": [0, 17.5],  "color": "#d5f5e3"},
                    {"range": [17.5, 22], "color": "#fdebd0"},
                    {"range": [22, 35],   "color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "value": NATIONAL_HF_READMIT}
            }
        ))
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Gauge: Average LOS vs national 5.4 days
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_los,
            title={"text": "Avg Length of Stay (days)"},
            delta={"reference": 5.4, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 15]},
                "bar": {"color": "#8e44ad"},
                "steps": [
                    {"range": [0, 4.5],  "color": "#d5f5e3"},
                    {"range": [4.5, 6],  "color": "#fdebd0"},
                    {"range": [6, 15],   "color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": "orange", "width": 3}, "value": 5.4}
            }
        ))
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # Gauge: Comorbidity burden
        avg_comorbidity = full_data['comorbidity_score'].mean()
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_comorbidity,
            title={"text": "Avg Comorbidity Score"},
            gauge={
                "axis": {"range": [0, 8]},
                "bar": {"color": "#e67e22"},
                "steps": [
                    {"range": [0, 2], "color": "#d5f5e3"},
                    {"range": [2, 4], "color": "#fdebd0"},
                    {"range": [4, 8], "color": "#fadbd8"},
                ]
            }
        ))
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Risk Stratification Summary ───────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Patient Risk Stratification Overview</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="story-banner">
    <strong>Clinical Insight:</strong> Risk stratification at discharge is the cornerstone of
    readmission prevention. CMS and Joint Commission guidelines recommend identifying patients
    in the <strong>High</strong> and <strong>Very High</strong> risk categories for structured
    post-discharge follow-up within 48–72 hours, medication reconciliation, and care coordinator
    assignment. The chart below shows your current patient risk distribution.
    </div>
    """, unsafe_allow_html=True)

    df = full_data.copy()
    df['risk_score'] = 0
    if 'age' in df.columns:              df['risk_score'] += (df['age'] > 75).astype(int) * 1
    if 'last_bnp' in df.columns:         df['risk_score'] += (df['last_bnp'] > 1000).astype(int) * 2
    if 'comorbidity_score' in df.columns:df['risk_score'] += (df['comorbidity_score'] >= 3).astype(int) * 2
    if 'length_of_stay' in df.columns:   df['risk_score'] += (df['length_of_stay'] > 7).astype(int) * 1
    if 'has_ckd' in df.columns:          df['risk_score'] += df['has_ckd'] * 1
    if 'has_copd' in df.columns:         df['risk_score'] += df['has_copd'] * 1
    if 'prior_admissions_6m' in df.columns: df['risk_score'] += (df['prior_admissions_6m'] > 0).astype(int) * 2

    df['risk_tier'] = pd.cut(df['risk_score'],
                              bins=[-1, 2, 4, 6, 100],
                              labels=['🟢 Low', '🟡 Moderate', '🟠 High', '🔴 Very High'])

    risk_summary = df.groupby('risk_tier', observed=True).agg(
        patients=('readmitted_30d', 'count'),
        readmit_rate=('readmitted_30d', 'mean')
    ).reset_index()
    risk_summary['readmit_rate'] = (risk_summary['readmit_rate'] * 100).round(1)
    risk_summary['intervention'] = [
        "Standard discharge instructions",
        "PCP follow-up within 7 days",
        "Case manager assigned + 48h phone call",
        "Intensive care transition program + home visit"
    ]

    col_a, col_b = st.columns([2, 3])

    with col_a:
        fig = px.pie(risk_summary, values='patients', names='risk_tier',
                     title="Patient Distribution by Risk Tier",
                     color_discrete_sequence=['#27ae60', '#f39c12', '#e67e22', '#e74c3c'],
                     hole=0.45)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = go.Figure()
        tier_colors = {'🟢 Low': '#27ae60', '🟡 Moderate': '#f39c12',
                       '🟠 High': '#e67e22', '🔴 Very High': '#e74c3c'}
        for _, row in risk_summary.iterrows():
            fig.add_trace(go.Bar(
                name=str(row['risk_tier']),
                x=[str(row['risk_tier'])],
                y=[row['readmit_rate']],
                marker_color=tier_colors.get(str(row['risk_tier']), '#888'),
                text=f"{row['readmit_rate']}%<br>({row['patients']:,} pts)",
                textposition='outside'
            ))
        fig.add_hline(y=NATIONAL_HF_READMIT, line_dash="dot",
                      line_color="red", annotation_text="National 22%")
        fig.update_layout(
            title="Readmission Rate by Risk Tier",
            yaxis_title="30-Day Readmission Rate (%)",
            yaxis_range=[0, max(risk_summary['readmit_rate'].max() * 1.3, 30)],
            showlegend=False, height=350, plot_bgcolor="white",
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0")
        )
        st.plotly_chart(fig, use_container_width=True)

    # Intervention table
    st.markdown("**📋 Recommended Interventions by Risk Tier**")
    st.dataframe(
        risk_summary.rename(columns={
            'risk_tier': 'Risk Tier', 'patients': 'Patients',
            'readmit_rate': 'Readmit Rate (%)', 'intervention': 'Recommended Intervention'
        }).style.format({'Patients': '{:,.0f}', 'Readmit Rate (%)': '{:.1f}%'}),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    # ── High-Risk Patient Alert List ──────────────────────────────────────────
    st.markdown('<div class="section-title">🚨 High-Risk Patient Alerts — Action Required</div>', unsafe_allow_html=True)

    high_risk = df[df['risk_tier'].isin(['🟠 High', '🔴 Very High'])].copy()
    display_cols = [c for c in ['patient_id', 'age', 'gender', 'comorbidity_score',
                                 'length_of_stay', 'last_bnp', 'discharge_location',
                                 'readmitted_30d', 'risk_score', 'risk_tier'] if c in high_risk.columns]
    high_risk_disp = high_risk[display_cols].sort_values('risk_score', ascending=False).head(20)

    if 'patient_id' in high_risk_disp.columns:
        high_risk_disp['patient_id'] = high_risk_disp['patient_id'].astype(str).str[:8] + "****"

    st.dataframe(
        high_risk_disp.style.map(
            lambda v: 'background-color: #fadbd8; font-weight:bold' if v == '🔴 Very High'
                      else ('background-color: #fdebd0' if v == '🟠 High' else ''),
            subset=['risk_tier'] if 'risk_tier' in high_risk_disp.columns else []
        ),
        use_container_width=True, height=350
    )

    csv = high_risk_disp.to_csv(index=False)
    st.download_button("📥 Download High-Risk Patient List", csv,
                       file_name="high_risk_patients.csv", mime="text/csv")
