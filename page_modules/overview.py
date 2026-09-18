"""
Clinical Overview & EDA — Enhanced US Healthcare Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def show(full_data, train_data, test_data):
    st.markdown("## 📊 Clinical Overview & Exploratory Data Analysis")

    st.markdown("""
    <div class="story-banner">
    <strong>About this section:</strong> This module provides a clinical deep-dive into the heart failure
    patient population. We examine demographics, lab values, comorbidities, and discharge patterns
    to understand <em>who</em> gets readmitted and <em>why</em> — using the same evidence-based
    lenses applied by US hospital quality improvement teams.
    </div>
    """, unsafe_allow_html=True)

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    readmit_rate = full_data['readmitted_30d'].mean() * 100
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Patients",     f"{len(full_data):,}")
    c2.metric("Readmission Rate",   f"{readmit_rate:.2f}%",
              delta=f"{readmit_rate - 22:.1f}% vs CMS avg", delta_color="inverse")
    c3.metric("Avg Age",            f"{full_data['age'].mean():.1f} yrs")
    c4.metric("Avg Length of Stay", f"{full_data['length_of_stay'].mean():.1f} days")
    c5.metric("Avg BNP",            f"{full_data['last_bnp'].mean():.0f} pg/mL")
    c6.metric("Comorbidity Score",  f"{full_data['comorbidity_score'].mean():.1f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distributions",
        "🔗 Correlations",
        "👥 Demographics",
        "🏥 Clinical Profile",
        "💊 Comorbidities"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Patient Population Distributions</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        Distribution analysis reveals the shape of our patient population.
        Note the <strong>age concentration in 65–80</strong> (Medicare population),
        and elevated BNP levels signaling decompensated heart failure — a key readmission driver.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            readmit_counts = full_data['readmitted_30d'].value_counts()
            labels = ['Not Readmitted (30d)', 'Readmitted (30d)']
            vals   = [readmit_counts.get(0.0, 0), readmit_counts.get(1.0, 0)]
            fig = go.Figure(go.Bar(
                x=labels, y=vals,
                marker_color=['#27ae60', '#e74c3c'],
                text=[f"{v:,}<br>({v/sum(vals)*100:.1f}%)" for v in vals],
                textposition='outside'
            ))
            fig.update_layout(
                title="30-Day Readmission Outcome",
                yaxis_title="Patients", height=360, plot_bgcolor="white",
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0")
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(full_data, x='age', color='readmitted_30d',
                               nbins=30,
                               title="Age Distribution by Readmission Status",
                               labels={'readmitted_30d': 'Readmitted', 'age': 'Age (years)'},
                               color_discrete_map={0.0: '#27ae60', 1.0: '#e74c3c'},
                               barmode='overlay', opacity=0.75)
            fig.add_vline(x=65, line_dash="dash", line_color="navy",
                          annotation_text="Medicare Eligibility (65)")
            fig.update_layout(height=360, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.violin(full_data, x='readmitted_30d', y='length_of_stay',
                            color='readmitted_30d', box=True,
                            title="Length of Stay Distribution",
                            labels={'readmitted_30d': 'Readmitted', 'length_of_stay': 'Days'},
                            color_discrete_map={0.0: '#27ae60', 1.0: '#e74c3c'})
            fig.add_hline(y=5.4, line_dash="dot", annotation_text="National Avg LOS 5.4d",
                          line_color="navy")
            fig.update_layout(height=360, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            bnp_clip = full_data.copy()
            bnp_clip['last_bnp'] = bnp_clip['last_bnp'].clip(upper=5000)
            fig = px.histogram(bnp_clip, x='last_bnp', color='readmitted_30d',
                               nbins=40, barmode='overlay', opacity=0.75,
                               title="BNP Levels (pg/mL) — Readmission Marker",
                               labels={'last_bnp': 'BNP (pg/mL)', 'readmitted_30d': 'Readmitted'},
                               color_discrete_map={0.0: '#27ae60', 1.0: '#e74c3c'})
            fig.add_vline(x=100, line_dash="dash", annotation_text="Normal <100",
                          line_color="green")
            fig.add_vline(x=400, line_dash="dash", annotation_text="HF Risk >400",
                          line_color="orange")
            fig.add_vline(x=1000, line_dash="dash", annotation_text="Severe >1000",
                          line_color="red")
            fig.update_layout(height=360, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        # Lab value distributions
        st.markdown('<div class="section-title">Key Lab Value Distributions</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="story-banner">
        Lab values like <strong>Creatinine</strong> (kidney function), <strong>Sodium</strong> (fluid balance),
        and <strong>Hemoglobin</strong> (anemia) are evidence-based readmission predictors validated
        in the ACC/AHA Heart Failure Guidelines.
        </div>
        """, unsafe_allow_html=True)

        lab_cols = [c for c in ['last_creatinine', 'last_sodium', 'last_hemoglobin'] if c in full_data.columns]
        lab_labels = {'last_creatinine': 'Creatinine (mg/dL)', 'last_sodium': 'Sodium (mEq/L)', 'last_hemoglobin': 'Hemoglobin (g/dL)'}
        lab_refs = {'last_creatinine': (0.7, 1.3), 'last_sodium': (136, 145), 'last_hemoglobin': (12, 17)}

        if lab_cols:
            fig = make_subplots(rows=1, cols=len(lab_cols),
                                subplot_titles=[lab_labels.get(c, c) for c in lab_cols])
            for i, col in enumerate(lab_cols):
                for val, color, name in [(0.0, '#27ae60', 'Not Readmitted'), (1.0, '#e74c3c', 'Readmitted')]:
                    subset = full_data[full_data['readmitted_30d'] == val][col].dropna()
                    fig.add_trace(go.Histogram(x=subset, name=name, opacity=0.7,
                                               marker_color=color, showlegend=(i == 0),
                                               nbinsx=30), row=1, col=i+1)
                lo, hi = lab_refs.get(col, (None, None))
                if lo:
                    fig.add_vline(x=lo, line_dash="dash", line_color="gray", row=1, col=i+1)
                    fig.add_vline(x=hi, line_dash="dash", line_color="gray", row=1, col=i+1)

            fig.update_layout(height=340, barmode='overlay', plot_bgcolor="white",
                              title_text="Lab Values by Readmission Status (gray dashes = normal range)")
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Feature Correlation Analysis</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        The correlation matrix reveals relationships between clinical variables.
        Strong positive correlations with <strong>readmitted_30d</strong> highlight the most
        actionable risk factors. Look for clustering patterns — comorbidities tend to travel together
        (e.g., CKD + diabetes + hypertension = cardiometabolic syndrome).
        </div>
        """, unsafe_allow_html=True)

        exclude = ['index_encounter_id', 'patient_id', 'days_to_readmit',
                   'index_admission', 'index_discharge', 'readmission_date']
        num_cols = [c for c in full_data.select_dtypes(include=[np.number]).columns if c not in exclude]
        corr = full_data[num_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu', zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 7},
            colorbar=dict(title="r")
        ))
        fig.update_layout(title="Feature Correlation Matrix", height=750,
                          xaxis={'side': 'bottom', 'tickangle': -45})
        st.plotly_chart(fig, use_container_width=True)

        # Top drivers
        st.markdown('<div class="section-title">Top Readmission Drivers (Correlation Strength)</div>', unsafe_allow_html=True)
        if 'readmitted_30d' in corr.columns:
            readmit_corr = corr['readmitted_30d'].drop('readmitted_30d').dropna()
            readmit_corr_sorted = readmit_corr.reindex(readmit_corr.abs().sort_values(ascending=False).index)
            top15 = readmit_corr_sorted.head(15)

            fig = go.Figure(go.Bar(
                x=top15.values, y=top15.index,
                orientation='h',
                marker_color=['#e74c3c' if v > 0 else '#3498db' for v in top15.values],
                text=[f"{v:.3f}" for v in top15.values], textposition='outside'
            ))
            fig.update_layout(
                title="Top 15 Features by Correlation with 30-Day Readmission",
                xaxis_title="Pearson Correlation Coefficient",
                height=450, plot_bgcolor="white",
                yaxis=dict(autorange="reversed"), xaxis=dict(gridcolor="#f0f0f0")
            )
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Demographic Analysis</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        <strong>Health Equity Lens:</strong> CMS and The Joint Commission require hospitals to track
        readmission rates by race/ethnicity, insurance type, and socioeconomic factors.
        Disparities in readmission rates across demographic groups often reflect gaps in access
        to follow-up care, transportation, and medication adherence — not just clinical acuity.
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            gender_g = full_data.groupby('gender')['readmitted_30d'].agg(['mean', 'count']).reset_index()
            gender_g['rate'] = gender_g['mean'] * 100
            fig = go.Figure(go.Bar(
                x=gender_g['gender'], y=gender_g['rate'],
                marker_color=['#1a6eb5', '#e91e8c', '#27ae60'][:len(gender_g)],
                text=[f"{r:.1f}%<br>({c:,} pts)" for r, c in zip(gender_g['rate'], gender_g['count'])],
                textposition='outside'
            ))
            fig.add_hline(y=22, line_dash="dot", annotation_text="CMS 22%", line_color="red")
            fig.update_layout(title="Readmission Rate by Gender", yaxis_title="%",
                              yaxis_range=[0, 30], height=360, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            race_g = full_data.groupby('race')['readmitted_30d'].agg(['mean', 'count']).reset_index()
            race_g['rate'] = race_g['mean'] * 100
            race_g = race_g.sort_values('rate', ascending=True)
            fig = go.Figure(go.Bar(
                x=race_g['rate'], y=race_g['race'],
                orientation='h',
                marker_color=['#e74c3c' if r > 22 else '#27ae60' for r in race_g['rate']],
                text=[f"{r:.1f}%" for r in race_g['rate']], textposition='outside'
            ))
            fig.add_vline(x=22, line_dash="dot", annotation_text="National Avg", line_color="red")
            fig.update_layout(title="Readmission Rate by Race/Ethnicity ⚖️ Health Equity",
                              xaxis_title="%", height=360, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            ins_g = full_data.groupby('insurance_type')['readmitted_30d'].agg(['mean', 'count']).reset_index()
            ins_g['rate'] = ins_g['mean'] * 100
            fig = px.bar(ins_g, x='insurance_type', y='rate',
                         color='rate', color_continuous_scale='RdYlGn_r',
                         title="Readmission Rate by Insurance Type",
                         text=[f"{r:.1f}%" for r in ins_g['rate']],
                         labels={'rate': 'Readmit Rate (%)', 'insurance_type': 'Insurance'})
            fig.add_hline(y=22, line_dash="dot", line_color="red")
            fig.update_traces(textposition='outside')
            fig.update_layout(height=360, plot_bgcolor="white", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            disc_g = full_data.groupby('discharge_location')['readmitted_30d'].agg(['mean', 'count']).reset_index()
            disc_g['rate'] = disc_g['mean'] * 100
            disc_g = disc_g.sort_values('rate', ascending=False)
            fig = px.bar(disc_g, x='discharge_location', y='rate',
                         color='rate', color_continuous_scale='Reds',
                         title="Readmission Rate by Discharge Destination",
                         text=[f"{r:.1f}%" for r in disc_g['rate']],
                         labels={'rate': 'Readmit Rate (%)', 'discharge_location': 'Discharge To'})
            fig.update_traces(textposition='outside')
            fig.update_layout(height=360, plot_bgcolor="white", coloraxis_showscale=False,
                              xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        # Age pyramid style view
        st.markdown('<div class="section-title">Age-Gender Population Pyramid</div>', unsafe_allow_html=True)
        age_bins = [0, 40, 50, 60, 65, 70, 75, 80, 85, 120]
        age_labels = ['<40', '40-49', '50-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+']
        df_ag = full_data.copy()
        df_ag['age_group'] = pd.cut(df_ag['age'], bins=age_bins, labels=age_labels)
        if 'gender' in df_ag.columns:
            pyr = df_ag.groupby(['age_group', 'gender'], observed=True).size().reset_index(name='count')
            male   = pyr[pyr['gender'].str.lower().str.contains('m', na=False)]
            female = pyr[pyr['gender'].str.lower().str.contains('f', na=False)]
            fig = go.Figure()
            fig.add_trace(go.Bar(y=male['age_group'].astype(str),   x=-male['count'],
                                 orientation='h', name='Male',   marker_color='#1a6eb5'))
            fig.add_trace(go.Bar(y=female['age_group'].astype(str), x=female['count'],
                                 orientation='h', name='Female', marker_color='#e91e8c'))
            fig.update_layout(
                title="Age-Gender Population Pyramid",
                barmode='relative', height=400, plot_bgcolor="white",
                xaxis=dict(title="← Male | Female →", tickvals=[], showgrid=False),
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">Clinical Profile Analysis</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        Clinical features validated by the <strong>ACC/AHA Heart Failure Society of America Guidelines</strong>
        as readmission predictors: elevated BNP, worsening renal function (creatinine), hyponatremia,
        anemia, and high comorbidity burden. Each metric below is compared between readmitted and
        non-readmitted patients.
        </div>
        """, unsafe_allow_html=True)

        clinical_features = [c for c in ['last_bnp', 'last_creatinine', 'last_sodium',
                                           'last_hemoglobin', 'comorbidity_score',
                                           'hf_med_count', 'total_meds', 'prior_admissions_6m']
                             if c in full_data.columns]

        # Multi-panel box plots
        n = len(clinical_features)
        rows, cols_per_row = (n + 3) // 4, 4
        fig = make_subplots(rows=rows, cols=cols_per_row,
                            subplot_titles=clinical_features)
        for idx, feat in enumerate(clinical_features):
            r, c = divmod(idx, cols_per_row)
            for val, color, name in [(0.0, '#27ae60', 'Not Readmitted'), (1.0, '#e74c3c', 'Readmitted')]:
                subset = full_data[full_data['readmitted_30d'] == val][feat].dropna()
                fig.add_trace(go.Box(y=subset, name=name, marker_color=color,
                                     showlegend=(idx == 0)), row=r+1, col=c+1)
        fig.update_layout(height=max(400, rows * 300), boxmode='group',
                          title_text="Clinical Features: Readmitted vs. Not Readmitted",
                          plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        # Scatter: BNP vs Creatinine (key interaction)
        if 'last_bnp' in full_data.columns and 'last_creatinine' in full_data.columns:
            st.markdown('<div class="section-title">BNP × Creatinine Interaction — Cardiorenal Syndrome</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="story-banner">
            <strong>Cardiorenal Syndrome</strong> occurs when heart failure worsens kidney function and vice versa.
            Patients with <em>both</em> elevated BNP and elevated creatinine are at the highest readmission risk.
            This scatter plot maps that clinical intersection.
            </div>
            """, unsafe_allow_html=True)

            fig = px.scatter(
                full_data.sample(min(2000, len(full_data))),
                x='last_creatinine', y='last_bnp',
                color='readmitted_30d',
                color_discrete_map={0.0: '#27ae60', 1.0: '#e74c3c'},
                opacity=0.6, size_max=6,
                title="Cardiorenal Risk Space: BNP vs. Creatinine",
                labels={'last_creatinine': 'Creatinine (mg/dL)', 'last_bnp': 'BNP (pg/mL)',
                        'readmitted_30d': 'Readmitted'}
            )
            fig.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="BNP >1000 (Severe HF)")
            fig.add_vline(x=1.3, line_dash="dash", line_color="orange", annotation_text="Cr >1.3 (CKD)")
            fig.update_layout(height=450, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-title">Comorbidity Burden & Readmission Impact</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        <strong>Comorbidity burden</strong> is the single strongest predictor of readmission.
        CMS risk-adjustment models (HCC — Hierarchical Condition Categories) account for comorbidities
        when calculating the Expected Readmission Rate. Understanding which conditions drive
        the most excess readmissions helps target intervention programs.
        </div>
        """, unsafe_allow_html=True)

        comorbidity_cols = [c for c in ['has_hypertension', 'has_diabetes', 'has_cad', 'has_copd',
                                         'has_ckd', 'has_afib', 'has_obesity', 'has_anemia']
                            if c in full_data.columns]

        impact = []
        for col in comorbidity_cols:
            n_with = full_data[col].sum()
            if n_with == 0:
                continue
            r_with    = full_data[full_data[col] == 1.0]['readmitted_30d'].mean() * 100
            r_without = full_data[full_data[col] == 0.0]['readmitted_30d'].mean() * 100
            excess    = r_with - r_without
            impact.append({
                'Condition': col.replace('has_', '').replace('_', ' ').title(),
                'Patients with Condition': int(n_with),
                'Prevalence (%)': round(n_with / len(full_data) * 100, 1),
                'Readmit Rate WITH': round(r_with, 2),
                'Readmit Rate WITHOUT': round(r_without, 2),
                'Excess Readmit Rate': round(excess, 2)
            })

        imp_df = pd.DataFrame(impact).sort_values('Excess Readmit Rate', ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(name='With Condition',    x=imp_df['Condition'], y=imp_df['Readmit Rate WITH'],    marker_color='#e74c3c'))
            fig.add_trace(go.Bar(name='Without Condition', x=imp_df['Condition'], y=imp_df['Readmit Rate WITHOUT'], marker_color='#27ae60'))
            fig.add_hline(y=22, line_dash="dot", annotation_text="National 22%", line_color="navy")
            fig.update_layout(
                title="Readmission Rate: With vs. Without Each Comorbidity",
                barmode='group', yaxis_title="Rate (%)", height=400,
                plot_bgcolor="white", xaxis_tickangle=-30
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(imp_df, x='Condition', y='Excess Readmit Rate',
                         color='Excess Readmit Rate', color_continuous_scale='Reds',
                         title="Excess Readmission Rate Attributable to Each Comorbidity",
                         text=[f"+{v:.1f}%" for v in imp_df['Excess Readmit Rate']])
            fig.update_traces(textposition='outside')
            fig.update_layout(height=400, plot_bgcolor="white",
                              coloraxis_showscale=False, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        # Comorbidity co-occurrence heatmap
        st.markdown('<div class="section-title">Comorbidity Co-occurrence Matrix</div>', unsafe_allow_html=True)
        co = full_data[comorbidity_cols].corr()
        co.columns = [c.replace('has_', '').upper() for c in co.columns]
        co.index   = [c.replace('has_', '').upper() for c in co.index]
        fig = go.Figure(go.Heatmap(
            z=co.values, x=co.columns, y=co.index,
            colorscale='Blues', zmid=0,
            text=np.round(co.values, 2), texttemplate='%{text}',
            textfont={"size": 9}
        ))
        fig.update_layout(title="Which Comorbidities Cluster Together?", height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.dataframe(
            imp_df.style.format({
                'Patients with Condition': '{:,.0f}',
                'Prevalence (%)': '{:.1f}%',
                'Readmit Rate WITH': '{:.2f}%',
                'Readmit Rate WITHOUT': '{:.2f}%',
                'Excess Readmit Rate': '{:+.2f}%'
            }).background_gradient(subset=['Excess Readmit Rate'], cmap='Reds'),
            use_container_width=True, hide_index=True
        )
