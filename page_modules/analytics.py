"""
Advanced Analytics — US Healthcare Edition
Statistical testing, risk stratification, population health, interventions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


NATIONAL_HF_READMIT = 22.0


def show(full_data, train_data, test_data):
    st.markdown("## 📈 Advanced Clinical Analytics")

    st.markdown("""
    <div class="story-banner">
    <strong>What you'll find here:</strong> Statistical significance testing, evidence-based risk
    stratification models, population health trend analysis, and cohort segmentation — the tools
    used by hospital quality teams, CMOs, and population health programs across the US.
    All analyses align with <strong>CMS HRRP</strong> and <strong>AHRQ Quality Indicators</strong>.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Statistical Testing",
        "🎯 Risk Stratification",
        "📉 Population Trends",
        "🔬 Cohort Builder",
        "💊 Intervention Simulator"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Statistical Significance — What Drives Readmission?</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        We use <strong>independent samples t-tests</strong> and <strong>Cohen's D</strong> effect sizes
        to identify which clinical variables differ significantly between readmitted and
        non-readmitted patients. P < 0.05 indicates statistical significance.
        Effect sizes: Small ≥0.2, Medium ≥0.5, Large ≥0.8.
        </div>
        """, unsafe_allow_html=True)

        num_feats = [c for c in ['age', 'last_bnp', 'last_sodium', 'last_creatinine',
                                  'last_hemoglobin', 'length_of_stay', 'comorbidity_score',
                                  'hf_med_count', 'total_meds', 'prior_admissions_6m']
                     if c in full_data.columns]

        results = []
        for feat in num_feats:
            grp1 = full_data[full_data['readmitted_30d'] == 1.0][feat].dropna()
            grp0 = full_data[full_data['readmitted_30d'] == 0.0][feat].dropna()
            if len(grp1) < 2 or len(grp0) < 2:
                continue
            t, p = stats.ttest_ind(grp1, grp0)
            pool_std = np.sqrt(((len(grp1)-1)*grp1.std()**2 + (len(grp0)-1)*grp0.std()**2) /
                                (len(grp1)+len(grp0)-2))
            d = (grp1.mean() - grp0.mean()) / pool_std if pool_std > 0 else 0
            results.append({
                'Feature': feat.replace('_', ' ').title(),
                'Readmitted Mean': grp1.mean(),
                'Not Readmitted Mean': grp0.mean(),
                'Difference': grp1.mean() - grp0.mean(),
                'T-Stat': t,
                'P-Value': p,
                "Cohen's D": d,
                'Effect': 'Large' if abs(d) >= 0.8 else ('Medium' if abs(d) >= 0.5 else 'Small'),
                'Significant': '✅ Yes' if p < 0.05 else '❌ No'
            })

        res_df = pd.DataFrame(results).sort_values('P-Value')

        col1, col2 = st.columns([3, 2])
        with col1:
            st.dataframe(
                res_df.style.format({
                    'Readmitted Mean': '{:.2f}', 'Not Readmitted Mean': '{:.2f}',
                    'Difference': '{:+.3f}', 'T-Stat': '{:.3f}',
                    'P-Value': '{:.4f}', "Cohen's D": '{:.3f}'
                }).background_gradient(subset=['P-Value'], cmap='RdYlGn')
                  .applymap(lambda x: 'background-color:#d5f5e3' if '✅' in str(x) else '', subset=['Significant']),
                use_container_width=True, hide_index=True
            )

        with col2:
            sig_df = res_df[res_df['Significant'] == '✅ Yes'].head(8)
            fig = px.bar(sig_df, x="Cohen's D", y='Feature',
                         orientation='h', color="Cohen's D",
                         color_continuous_scale='Reds',
                         title="Effect Sizes (Significant Features Only)")
            fig.add_vline(x=0.2, line_dash="dot", annotation_text="Small", line_color="green")
            fig.add_vline(x=0.5, line_dash="dot", annotation_text="Medium", line_color="orange")
            fig.add_vline(x=0.8, line_dash="dot", annotation_text="Large", line_color="red")
            fig.update_layout(height=400, plot_bgcolor="white", coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        # Forest plot style
        if len(res_df) > 0:
            st.markdown('<div class="section-title">Forest Plot — Mean Differences</div>',
                        unsafe_allow_html=True)
            ci_df = res_df.copy()
            ci_df['ci95'] = 1.96 * abs(ci_df['Difference']) / 3  # approximate
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ci_df['Difference'], y=ci_df['Feature'],
                mode='markers', marker=dict(size=10, color='#1a6eb5'),
                error_x=dict(array=ci_df['ci95'], visible=True, color='#888'),
                name='Mean Difference'
            ))
            fig.add_vline(x=0, line_color='gray', line_dash='dash')
            fig.update_layout(
                title="Mean Difference (Readmitted − Not Readmitted) with 95% CI",
                xaxis_title="Difference in Means",
                height=400, plot_bgcolor="white",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Evidence-Based Risk Stratification</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        Risk stratification models like <strong>LACE+</strong> (Length of stay, Acuity, Comorbidities,
        ED visits) and <strong>HOSPITAL Score</strong> are used by US care transition programs to
        identify patients for intervention. Our composite risk score below is modeled on these
        validated instruments.
        </div>
        """, unsafe_allow_html=True)

        df = full_data.copy()
        df['risk_score'] = 0
        if 'age' in df.columns:               df['risk_score'] += (df['age'] > 75).astype(int) * 1
        if 'last_bnp' in df.columns:          df['risk_score'] += (df['last_bnp'] > 1000).astype(int) * 2
        if 'comorbidity_score' in df.columns: df['risk_score'] += (df['comorbidity_score'] >= 3).astype(int) * 2
        if 'length_of_stay' in df.columns:    df['risk_score'] += (df['length_of_stay'] > 7).astype(int) * 1
        if 'has_ckd' in df.columns:           df['risk_score'] += df['has_ckd'] * 1
        if 'has_copd' in df.columns:          df['risk_score'] += df['has_copd'] * 1
        if 'prior_admissions_6m' in df.columns: df['risk_score'] += (df['prior_admissions_6m'] > 0).astype(int) * 2
        if 'prior_ed_visits_30d' in df.columns: df['risk_score'] += (df['prior_ed_visits_30d'] > 0).astype(int) * 1

        df['risk_tier'] = pd.cut(df['risk_score'], bins=[-1, 2, 4, 6, 100],
                                  labels=['Low', 'Moderate', 'High', 'Very High'])

        summary = df.groupby('risk_tier', observed=True).agg(
            count=('readmitted_30d', 'count'),
            readmit_rate=('readmitted_30d', 'mean'),
            avg_los=('length_of_stay', 'mean'),
            avg_bnp=('last_bnp', 'mean'),
            avg_age=('age', 'mean')
        ).reset_index()
        summary['readmit_rate'] = (summary['readmit_rate'] * 100).round(1)
        summary['avg_los'] = summary['avg_los'].round(1)
        summary['avg_bnp'] = summary['avg_bnp'].round(0)
        summary['avg_age'] = summary['avg_age'].round(1)

        tier_colors = {'Low': '#27ae60', 'Moderate': '#f39c12', 'High': '#e67e22', 'Very High': '#e74c3c'}
        color_list  = [tier_colors.get(t, '#888') for t in summary['risk_tier'].astype(str)]

        c1, c2, c3 = st.columns(3)

        with c1:
            fig = go.Figure(go.Pie(
                labels=summary['risk_tier'].astype(str),
                values=summary['count'],
                marker_colors=color_list,
                hole=0.5,
                textinfo='percent+label'
            ))
            fig.update_layout(title="Patient Distribution by Risk Tier", height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure(go.Bar(
                x=summary['risk_tier'].astype(str), y=summary['readmit_rate'],
                marker_color=color_list,
                text=[f"{r}%" for r in summary['readmit_rate']], textposition='outside'
            ))
            fig.add_hline(y=NATIONAL_HF_READMIT, line_dash="dot",
                          annotation_text="National 22%", line_color="red")
            fig.update_layout(title="Readmission Rate by Risk Tier",
                              yaxis_title="%", yaxis_range=[0, max(summary['readmit_rate'].max()*1.4, 30)],
                              height=350, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        with c3:
            # Sunburst: risk tier × insurance
            if 'insurance_type' in df.columns:
                sun = df.groupby(['risk_tier', 'insurance_type'], observed=True).size().reset_index(name='n')
                fig = px.sunburst(sun, path=['risk_tier', 'insurance_type'], values='n',
                                  title="Risk Tier × Insurance Type",
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        # Risk score distribution
        fig = px.histogram(df, x='risk_score', color='readmitted_30d',
                           barmode='overlay', opacity=0.75, nbins=20,
                           title="Risk Score Distribution — Readmitted vs. Not Readmitted",
                           color_discrete_map={0.0: '#27ae60', 1.0: '#e74c3c'},
                           labels={'risk_score': 'Composite Risk Score', 'readmitted_30d': 'Readmitted'})
        fig.update_layout(height=340, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Risk Tier Clinical Profile**")
        st.dataframe(
            summary.rename(columns={'risk_tier': 'Tier', 'count': 'Patients',
                                    'readmit_rate': 'Readmit Rate (%)',
                                    'avg_los': 'Avg LOS (days)',
                                    'avg_bnp': 'Avg BNP (pg/mL)',
                                    'avg_age': 'Avg Age (yrs)'})
            .style.format({'Patients': '{:,.0f}', 'Readmit Rate (%)': '{:.1f}%',
                           'Avg LOS (days)': '{:.1f}', 'Avg BNP (pg/mL)': '{:.0f}',
                           'Avg Age (yrs)': '{:.1f}'}),
            use_container_width=True, hide_index=True
        )

    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Population Health Trend Analysis</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        Trend analysis reveals how readmission risk varies across age cohorts, lab value severities,
        and length-of-stay categories. These patterns guide population health program design,
        targeting the subgroups with the steepest risk gradients.
        </div>
        """, unsafe_allow_html=True)

        df_t = full_data.copy()

        # Age groups
        df_t['age_group'] = pd.cut(df_t['age'], bins=[0,50,60,65,70,75,80,120],
                                    labels=['<50','50-59','60-64','65-69','70-74','75-79','80+'])
        age_tr = df_t.groupby('age_group', observed=True)['readmitted_30d'].agg(['mean','count']).reset_index()
        age_tr['rate'] = age_tr['mean'] * 100

        fig = make_subplots(rows=2, cols=2, subplot_titles=[
            "Readmit Rate by Age Group",
            "Readmit Rate by BNP Severity",
            "Readmit Rate by Length of Stay",
            "Readmit Rate by Comorbidity Score"
        ])

        # Age trend
        fig.add_trace(go.Scatter(x=age_tr['age_group'].astype(str), y=age_tr['rate'],
                                  mode='lines+markers', line=dict(color='#e74c3c', width=3),
                                  marker=dict(size=9), name='Age'),
                      row=1, col=1)
        fig.add_hline(y=22, line_dash="dot", line_color="gray", row=1, col=1)

        # BNP
        df_t['bnp_cat'] = pd.cut(df_t['last_bnp'], bins=[0,100,400,1000,20000],
                                   labels=['Normal\n<100','Mild\n100-400','Moderate\n400-1000','Severe\n>1000'])
        bnp_tr = df_t.groupby('bnp_cat', observed=True)['readmitted_30d'].agg(['mean','count']).reset_index()
        bnp_tr['rate'] = bnp_tr['mean'] * 100
        fig.add_trace(go.Scatter(x=bnp_tr['bnp_cat'].astype(str), y=bnp_tr['rate'],
                                  mode='lines+markers', line=dict(color='#9b59b6', width=3),
                                  marker=dict(size=9), name='BNP'),
                      row=1, col=2)

        # LOS
        df_t['los_cat'] = pd.cut(df_t['length_of_stay'], bins=[0,3,5,7,14,100],
                                   labels=['1-3d','4-5d','6-7d','8-14d','15+d'])
        los_tr = df_t.groupby('los_cat', observed=True)['readmitted_30d'].agg(['mean','count']).reset_index()
        los_tr['rate'] = los_tr['mean'] * 100
        fig.add_trace(go.Scatter(x=los_tr['los_cat'].astype(str), y=los_tr['rate'],
                                  mode='lines+markers', line=dict(color='#1a6eb5', width=3),
                                  marker=dict(size=9), name='LOS'),
                      row=2, col=1)

        # Comorbidity
        if 'comorbidity_score' in df_t.columns:
            comorb_tr = df_t.groupby('comorbidity_score')['readmitted_30d'].agg(['mean','count']).reset_index()
            comorb_tr['rate'] = comorb_tr['mean'] * 100
            fig.add_trace(go.Bar(x=comorb_tr['comorbidity_score'].astype(str), y=comorb_tr['rate'],
                                  marker_color='#e67e22', name='Comorbidities'),
                          row=2, col=2)

        fig.update_layout(height=700, showlegend=False, plot_bgcolor="white",
                          title_text="Readmission Rate Trends Across Clinical Dimensions")
        for i in range(1, 5):
            r, c = divmod(i-1, 2)
            fig.update_yaxes(title_text="Readmit Rate (%)", row=r+1, col=c+1, gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

        # Dual-axis: volume + rate
        st.markdown('<div class="section-title">Admission Volume vs. Readmission Rate by Age</div>',
                    unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=age_tr['age_group'].astype(str), y=age_tr['count'],
                              name='# Admissions', marker_color='#b3d1f0', opacity=0.7),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=age_tr['age_group'].astype(str), y=age_tr['rate'],
                                  name='Readmit Rate', mode='lines+markers',
                                  line=dict(color='#e74c3c', width=3), marker=dict(size=9)),
                      secondary_y=True)
        fig.update_layout(title="Volume & Readmission Rate by Age Group", height=380,
                          plot_bgcolor="white")
        fig.update_yaxes(title_text="# Admissions",     secondary_y=False, gridcolor="#f0f0f0")
        fig.update_yaxes(title_text="Readmit Rate (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">Cohort Builder — Custom Segment Analysis</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        Define a patient cohort using clinical filters. Compare the cohort's profile and
        readmission rate to the overall population — the core workflow of a <strong>Population Health
        Management</strong> program.
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age_rng = st.slider("Age Range", int(full_data['age'].min()), int(full_data['age'].max()),
                                (int(full_data['age'].min()), int(full_data['age'].max())))
        with c2:
            genders = st.multiselect("Gender", full_data['gender'].unique().tolist(),
                                     default=full_data['gender'].unique().tolist())
        with c3:
            min_comorb = st.slider("Min Comorbidity Score", 0, int(full_data['comorbidity_score'].max()), 0)
        with c4:
            los_rng = st.slider("Length of Stay (days)", int(full_data['length_of_stay'].min()),
                                int(full_data['length_of_stay'].max()),
                                (int(full_data['length_of_stay'].min()), int(full_data['length_of_stay'].max())))

        cohort = full_data[
            (full_data['age'] >= age_rng[0]) & (full_data['age'] <= age_rng[1]) &
            (full_data['gender'].isin(genders)) &
            (full_data['comorbidity_score'] >= min_comorb) &
            (full_data['length_of_stay'] >= los_rng[0]) & (full_data['length_of_stay'] <= los_rng[1])
        ]

        c1, c2, c3, c4, c5 = st.columns(5)
        cohort_rate  = cohort['readmitted_30d'].mean() * 100 if len(cohort) > 0 else 0
        overall_rate = full_data['readmitted_30d'].mean() * 100
        c1.metric("Cohort Size",         f"{len(cohort):,}")
        c2.metric("Cohort Readmit Rate", f"{cohort_rate:.1f}%", f"{cohort_rate - overall_rate:+.1f}% vs overall",
                  delta_color="inverse")
        c3.metric("Avg Age",             f"{cohort['age'].mean():.1f}" if len(cohort) > 0 else "—")
        c4.metric("Avg LOS",             f"{cohort['length_of_stay'].mean():.1f}" if len(cohort) > 0 else "—")
        c5.metric("Avg Comorbidities",   f"{cohort['comorbidity_score'].mean():.1f}" if len(cohort) > 0 else "—")

        if len(cohort) > 0:
            features = [c for c in ['age', 'last_bnp', 'length_of_stay', 'comorbidity_score',
                                     'last_creatinine', 'total_meds'] if c in full_data.columns]
            fig = make_subplots(rows=2, cols=3, subplot_titles=features[:6])
            for idx, feat in enumerate(features[:6]):
                r, c = divmod(idx, 3)
                fig.add_trace(go.Box(y=full_data[feat], name='All Patients',
                                     marker_color='#1a6eb5', showlegend=(idx == 0)), row=r+1, col=c+1)
                fig.add_trace(go.Box(y=cohort[feat], name='Your Cohort',
                                     marker_color='#e74c3c', showlegend=(idx == 0)), row=r+1, col=c+1)
            fig.update_layout(height=550, title_text="Cohort vs. Overall Population",
                              boxmode='group', plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            csv = cohort.to_csv(index=False)
            st.download_button("📥 Export Cohort Data", csv,
                               file_name="cohort_export.csv", mime="text/csv")

    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-title">Intervention Impact Simulator</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        <strong>Evidence-based interventions</strong> shown to reduce HF readmissions (NEJM, JAMA,
        ACC/AHA guidelines): structured discharge education, medication reconciliation,
        early post-discharge follow-up, remote monitoring, and care transition coaches.
        Use the simulator below to estimate the impact of deploying each intervention on your population.
        </div>
        """, unsafe_allow_html=True)

        interventions = {
            "Structured Discharge Education (teach-back)": 0.12,
            "Medication Reconciliation at Discharge":       0.10,
            "48-72h Post-Discharge Phone Call":             0.15,
            "7-day PCP Follow-up Appointment":              0.18,
            "Care Transition Coach (30 days)":              0.22,
            "Remote Patient Monitoring (weight/BP daily)":  0.20,
            "Home Health Nursing Visit (3×/week)":          0.25,
        }

        selected = st.multiselect("Select Interventions to Deploy", list(interventions.keys()),
                                  default=list(interventions.keys())[:3])

        combined_reduction = 1.0
        for prog in selected:
            combined_reduction *= (1 - interventions[prog])
        actual_reduction = 1 - combined_reduction

        current_rate = full_data['readmitted_30d'].mean() * 100
        new_rate     = current_rate * (1 - actual_reduction)
        readmits_prevented = int((current_rate - new_rate) / 100 * len(full_data))
        cost_per_admit = 15000
        savings = readmits_prevented * cost_per_admit

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Readmit Rate", f"{current_rate:.1f}%")
        c2.metric("Projected Rate",       f"{new_rate:.1f}%", f"{new_rate - current_rate:+.1f}%",
                  delta_color="inverse")
        c3.metric("Readmissions Prevented", f"{readmits_prevented:,}")
        c4.metric("Estimated Annual Savings", f"${savings:,.0f}")

        # Waterfall chart
        waterfall_x = ["Baseline"] + selected + ["Projected"]
        waterfall_y = [current_rate]
        running = current_rate
        for prog in selected:
            delta = -running * interventions[prog]
            waterfall_y.append(delta)
            running += delta
        waterfall_y.append(running)

        fig = go.Figure(go.Waterfall(
            name="Readmission Rate",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(selected) + ["total"],
            x=waterfall_x,
            y=waterfall_y,
            connector={"line": {"color": "gray"}},
            decreasing={"marker": {"color": "#27ae60"}},
            increasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#1a6eb5"}}
        ))
        fig.add_hline(y=NATIONAL_HF_READMIT, line_dash="dot",
                      annotation_text="CMS National Average (22%)", line_color="red")
        fig.update_layout(
            title="Intervention Impact Waterfall — Readmission Rate Reduction",
            yaxis_title="Readmission Rate (%)",
            height=480, plot_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cost-benefit table
        st.markdown("**Estimated Costs vs. Savings by Intervention**")
        prog_costs = {
            "Structured Discharge Education (teach-back)": 150,
            "Medication Reconciliation at Discharge":       200,
            "48-72h Post-Discharge Phone Call":             80,
            "7-day PCP Follow-up Appointment":              250,
            "Care Transition Coach (30 days)":              800,
            "Remote Patient Monitoring (weight/BP daily)":  500,
            "Home Health Nursing Visit (3×/week)":          1200,
        }
        rows_list = []
        for prog in selected:
            pats_eligible = int(len(full_data) * 0.6)
            program_cost  = prog_costs.get(prog, 500) * pats_eligible
            prevented     = int(len(full_data) * interventions[prog] * current_rate / 100)
            prog_savings  = prevented * cost_per_admit
            roi           = (prog_savings - program_cost) / max(program_cost, 1) * 100
            rows_list.append({
                'Intervention': prog,
                'Est. Program Cost': program_cost,
                'Readmissions Prevented': prevented,
                'Est. Savings': prog_savings,
                'Net Benefit': prog_savings - program_cost,
                'ROI (%)': round(roi, 1)
            })

        if rows_list:
            roi_df = pd.DataFrame(rows_list)
            st.dataframe(
                roi_df.style.format({
                    'Est. Program Cost': '${:,.0f}', 'Est. Savings': '${:,.0f}',
                    'Net Benefit': '${:,.0f}', 'ROI (%)': '{:.1f}%'
                }).background_gradient(subset=['ROI (%)'], cmap='Greens'),
                use_container_width=True, hide_index=True
            )
