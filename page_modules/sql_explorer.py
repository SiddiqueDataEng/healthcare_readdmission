"""
SQL Intelligence Explorer — US Healthcare Edition
Ad-hoc queries + pre-built clinical analytics with narrative context
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from streamlit_ace import st_ace

sys.path.append(str(Path(__file__).parent.parent))
from src.sql_analytics import SQLAnalytics


def convert_types(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)
        elif hasattr(df[col].dtype, 'numpy_dtype'):
            df[col] = df[col].astype(float)
    return df


def show():
    st.markdown("## 🔬 SQL Intelligence Explorer")

    st.markdown("""
    <div class="story-banner">
    <strong>Why SQL in healthcare analytics?</strong> Clinical data lives in relational databases —
    EHR tables (patients, encounters, diagnoses, labs, medications) mirror systems like Epic Clarity,
    Cerner CareAware, and Health Catalyst. SQL lets quality teams and analysts query the raw data
    behind every metric, validating calculations and answering ad-hoc questions without waiting
    for IT. The EHR schema below mirrors a simplified <strong>OMOP CDM</strong> structure.
    </div>
    """, unsafe_allow_html=True)

    # Initialize connection
    if 'sql_analytics' not in st.session_state:
        st.session_state.sql_analytics = SQLAnalytics()
        st.session_state.sql_analytics.connect()
    analytics = st.session_state.sql_analytics

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Custom Query",
        "🏥 Clinical Analytics Library",
        "📋 SQL Patterns Reference",
        "📈 Query History"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Write & Execute SQL Queries</div>', unsafe_allow_html=True)

        col_q, col_info = st.columns([3, 1])

        with col_info:
            st.markdown("""
            **📂 EHR Schema**

            `patients`
            — patient_id, age, gender, race, insurance_type

            `encounters`
            — encounter_id, patient_id, admission_date, discharge_date, length_of_stay, discharge_location, readmitted_30d

            `diagnoses`
            — diagnosis_id, encounter_id, icd_code, diagnosis_name, is_primary

            `labs`
            — lab_id, encounter_id, lab_name, lab_value, lab_date

            `medications`
            — med_id, encounter_id, medication_name, dose, frequency
            """)
            if st.button("🗂️ Show Table Sizes"):
                try:
                    info = analytics.get_table_info()
                    st.dataframe(info, use_container_width=True)
                except Exception as e:
                    st.error(str(e))

        with col_q:
            default_q = "SELECT p.gender, COUNT(*) as patients,\n       AVG(JULIANDAY(e.discharge_date) - JULIANDAY(e.admission_date)) as avg_los,\n       AVG(e.is_heart_failure) * 100 as heart_failure_pct\nFROM patients p\nJOIN encounters e ON p.patient_id = e.patient_id\nGROUP BY p.gender\nORDER BY heart_failure_pct DESC"
            query = st_ace(
                value=default_q,
                language="sql",
                theme="tomorrow_night",
                height=200,
                font_size=14,
                wrap=True,
                show_gutter=True,
                key="sql_query_editor",
            )

            col_a, col_b = st.columns([1, 3])
            with col_a:
                run = st.button("▶️ Run Query", type="primary", use_container_width=True)
            with col_b:
                limit = st.number_input("Row limit", 10, 10000, 500)

        if run and query:
            try:
                with st.spinner("Executing..."):
                    result = analytics.execute_query(query if 'LIMIT' in query.upper()
                                                     else query + f" LIMIT {limit}")
                st.success(f"✅ {len(result):,} rows returned")

                st.dataframe(result, use_container_width=True, height=380)

                st.download_button("📥 Download CSV", result.to_csv(index=False),
                                   file_name="query_result.csv", mime="text/csv")

                # Auto-viz
                if len(result) > 0:
                    with st.expander("📊 Visualize Results", expanded=True):
                        num_cols = result.select_dtypes(include='number').columns.tolist()
                        all_cols = result.columns.tolist()

                        cv1, cv2, cv3, cv4 = st.columns(4)
                        with cv1: viz = st.selectbox("Chart type", ["Bar","Line","Scatter","Pie","Histogram","Box"])
                        with cv2: xc = st.selectbox("X axis", all_cols)
                        with cv3: yc = st.selectbox("Y axis", num_cols or all_cols)
                        with cv4: cc = st.selectbox("Color", ["None"] + all_cols)

                        color = None if cc == "None" else cc
                        rp = convert_types(result)
                        try:
                            if viz == "Bar":        fig = px.bar(rp, x=xc, y=yc, color=color)
                            elif viz == "Line":     fig = px.line(rp, x=xc, y=yc, color=color)
                            elif viz == "Scatter":  fig = px.scatter(rp, x=xc, y=yc, color=color)
                            elif viz == "Pie":      fig = px.pie(rp, names=xc, values=yc)
                            elif viz == "Histogram":fig = px.histogram(rp, x=xc, color=color)
                            elif viz == "Box":      fig = px.box(rp, x=xc, y=yc, color=color)
                            fig.update_layout(height=420, plot_bgcolor="white")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as ve:
                            st.error(f"Visualization error: {ve}")

            except Exception as e:
                st.error(f"❌ Query error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Clinical Analytics Library — Pre-built Queries</div>',
                    unsafe_allow_html=True)

        analyses = {
            "🏥 Patient Cohort & Encounter Summary": {
                "desc": "Segment patients by encounter frequency quartile. Heavy utilizers (top quartile) account for ~70% of costs in most US health systems.",
                "fn": "patient_cohort_analysis",
                "viz": "bar", "x": "encounter_quartile", "y": "patient_count"
            },
            "📉 30/90-Day Readmission Trends": {
                "desc": "Track readmission rates over time — required for CMS quality reporting and HRRP monitoring. Seasonal spikes often coincide with flu season (Nov–Feb).",
                "fn": "readmission_trend_analysis",
                "viz": "line_dual", "x": "discharge_month"
            },
            "🧪 Lab Value Longitudinal Trends": {
                "desc": "Deteriorating lab values (rising BNP, rising creatinine) in the 72h before discharge strongly predict readmission per ACC/AHA guidelines.",
                "fn": "lab_value_trends",
                "viz": "line_lab"
            },
            "💊 Medication Adherence Analysis": {
                "desc": "Medication non-adherence causes ~25% of preventable HF readmissions. Diuretics (furosemide), ACE inhibitors, and beta-blockers are the critical drug classes.",
                "fn": "medication_adherence_analysis",
                "viz": "bar_stacked", "x": "medication_name", "y": "patient_count", "color": "adherence_category"
            },
            "🔀 Diagnosis Co-occurrence Patterns": {
                "desc": "Comorbidity clustering (e.g., HF + CKD + anemia = 'triple whammy') identifies high-complexity patients who need multi-disciplinary care management.",
                "fn": "diagnosis_co_occurrence",
                "viz": "bar_h", "x": "patient_count", "y": "pair"
            },
            "📏 Length of Stay by Decile": {
                "desc": "LOS outliers (top decile) drive disproportionate costs and are high readmission risk. SNF/skilled nursing placement at discharge is a key LOS driver.",
                "fn": "length_of_stay_analysis",
                "viz": "bar", "x": "los_decile", "y": "avg_los"
            },
            "🚨 High-Risk Patient Identification": {
                "desc": "Multi-criteria risk identification mirrors LACE+ score logic used by case managers. Flags patients needing care transition program enrollment.",
                "fn": "high_risk_patient_identification",
                "viz": "pie", "x": "risk_category"
            },
            "🗓️ Seasonal Admission Patterns": {
                "desc": "Seasonal variation guides staffing, bed management, and pre-emptive outreach scheduling. Heart failure exacerbations spike in winter (cold weather, holiday diet).",
                "fn": "seasonal_admission_patterns",
                "viz": "bar_group", "x": "season", "y": "admission_count", "color": "day_type"
            },
        }

        chosen = st.selectbox("Select Clinical Analysis", list(analyses.keys()))
        meta = analyses[chosen]

        st.markdown(f"""
        <div class="story-banner">
        <strong>Clinical Context:</strong> {meta['desc']}
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner(f"Running {chosen}..."):
                try:
                    fn = getattr(analytics, meta['fn'])
                    result = fn()
                    st.success(f"✅ {len(result):,} rows returned")

                    col_t, col_v = st.columns([2, 3])
                    with col_t:
                        st.dataframe(result.head(30), use_container_width=True, height=400)
                        st.download_button("📥 Download", result.to_csv(index=False),
                                           file_name=f"{meta['fn']}.csv", mime="text/csv")

                    with col_v:
                        rp = convert_types(result)
                        viz = meta['viz']

                        try:
                            if viz == "bar":
                                fig = px.bar(rp, x=meta['x'], y=meta['y'],
                                             title=chosen, color=meta['y'],
                                             color_continuous_scale='Blues')
                                fig.update_layout(coloraxis_showscale=False)

                            elif viz == "line_dual":
                                fig = go.Figure()
                                if 'readmission_rate_30d' in rp.columns:
                                    fig.add_trace(go.Scatter(x=rp[meta['x']],
                                                              y=rp['readmission_rate_30d'],
                                                              mode='lines+markers', name='30-Day Rate'))
                                if 'readmission_rate_90d' in rp.columns:
                                    fig.add_trace(go.Scatter(x=rp[meta['x']],
                                                              y=rp['readmission_rate_90d'],
                                                              mode='lines+markers', name='90-Day Rate'))
                                fig.update_layout(title="Readmission Rates Over Time",
                                                  xaxis_title="Month", yaxis_title="Rate (%)")

                            elif viz == "line_lab":
                                if 'lab_name' in rp.columns:
                                    labs = rp['lab_name'].unique()
                                    lab_sel = st.selectbox("Select Lab", labs)
                                    ld = rp[rp['lab_name'] == lab_sel].head(30).reset_index(drop=True)
                                    ld['seq'] = range(len(ld))
                                    fig = px.line(ld, x='seq', y='current_value',
                                                  title=f"{lab_sel} — Longitudinal Trend",
                                                  markers=True)
                                else:
                                    fig = px.line(rp, title=chosen)

                            elif viz == "bar_stacked":
                                fig = px.bar(rp, x=meta['x'], y=meta['y'], color=meta.get('color'),
                                             title=chosen, barmode='stack')
                                fig.update_xaxes(tickangle=-40)

                            elif viz == "bar_h":
                                if 'diagnosis_1' in rp.columns and 'diagnosis_2' in rp.columns:
                                    rp = rp.copy()
                                    rp['pair'] = rp['diagnosis_1'] + ' + ' + rp['diagnosis_2']
                                top = rp.head(10)
                                fig = px.bar(top, x=meta['x'], y='pair' if 'pair' in top.columns else meta.get('y', top.columns[0]),
                                             orientation='h', title=f"Top 10 {chosen}",
                                             color=meta['x'], color_continuous_scale='Reds')
                                fig.update_layout(coloraxis_showscale=False,
                                                  yaxis=dict(autorange="reversed"))

                            elif viz == "pie":
                                counts = rp[meta['x']].value_counts().reset_index()
                                counts.columns = ['category', 'count']
                                fig = px.pie(counts, names='category', values='count',
                                             title=chosen, hole=0.4)

                            elif viz == "bar_group":
                                fig = px.bar(rp, x=meta['x'], y=meta['y'],
                                             color=meta.get('color'),
                                             title=chosen, barmode='group')

                            else:
                                fig = px.bar(rp.head(20), title=chosen)

                            fig.update_layout(height=450, plot_bgcolor="white")
                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as ve:
                            st.warning(f"Chart error: {ve}")
                            st.dataframe(result, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Healthcare SQL Patterns Reference</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        These SQL patterns mirror queries used by analysts in Epic Clarity, Cerner Analytics,
        and Health Catalyst. Understanding them is foundational to healthcare data science.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔹 Index Admission Logic (First vs. Subsequent)"):
            st.markdown("**Use case:** Identify the first qualifying admission for a 30-day readmission cohort (mimics CMS methodology)")
            st.code("""
WITH ranked_encounters AS (
    SELECT
        patient_id,
        encounter_id,
        admission_date,
        discharge_date,
        ROW_NUMBER() OVER (
            PARTITION BY patient_id
            ORDER BY admission_date
        ) AS visit_rank
    FROM encounters
    WHERE discharge_date IS NOT NULL
),
index_admissions AS (
    SELECT * FROM ranked_encounters WHERE visit_rank = 1
),
readmission_check AS (
    SELECT
        ia.patient_id,
        ia.encounter_id AS index_encounter,
        ia.discharge_date AS index_discharge,
        MIN(e.admission_date) AS readmit_date,
        DATEDIFF(MIN(e.admission_date), ia.discharge_date) AS days_to_readmit
    FROM index_admissions ia
    LEFT JOIN encounters e
        ON ia.patient_id = e.patient_id
        AND e.admission_date > ia.discharge_date
        AND e.admission_date <= DATE(ia.discharge_date, '+30 days')
    GROUP BY ia.patient_id, ia.encounter_id
)
SELECT
    r.*,
    CASE WHEN readmit_date IS NOT NULL THEN 1 ELSE 0 END AS readmitted_30d
FROM readmission_check r;
            """, language='sql')

        with st.expander("🔹 LACE Score Calculation"):
            st.markdown("**Use case:** Calculate LACE readmission risk score (Length of stay, Acuity, Comorbidities, ED visits) — validated to predict 30-day readmission")
            st.code("""
WITH lace_components AS (
    SELECT
        e.patient_id,
        e.encounter_id,
        -- L: Length of stay (0-7 points)
        CASE
            WHEN e.length_of_stay <= 1  THEN 1
            WHEN e.length_of_stay <= 2  THEN 2
            WHEN e.length_of_stay <= 3  THEN 3
            WHEN e.length_of_stay <= 6  THEN 4
            WHEN e.length_of_stay <= 13 THEN 5
            ELSE 7
        END AS lace_l,
        -- A: Acuity (emergency admission = 3)
        3 AS lace_a,
        -- C: Comorbidities (Charlson-based, simplified)
        CASE
            WHEN p.comorbidity_score >= 5 THEN 5
            WHEN p.comorbidity_score >= 3 THEN 3
            WHEN p.comorbidity_score >= 1 THEN 1
            ELSE 0
        END AS lace_c,
        -- E: ED visits in prior 6 months
        COALESCE(ed.ed_visits, 0) AS lace_e
    FROM encounters e
    JOIN patients p ON e.patient_id = p.patient_id
    LEFT JOIN (
        SELECT patient_id, COUNT(*) as ed_visits
        FROM encounters
        WHERE admission_type = 'ED'
        GROUP BY patient_id
    ) ed ON e.patient_id = ed.patient_id
)
SELECT
    patient_id, encounter_id,
    lace_l + lace_a + lace_c + lace_e AS lace_score,
    CASE
        WHEN lace_l + lace_a + lace_c + lace_e >= 10 THEN 'High Risk'
        WHEN lace_l + lace_a + lace_c + lace_e >= 7  THEN 'Moderate Risk'
        ELSE 'Low Risk'
    END AS risk_category
FROM lace_components;
            """, language='sql')

        with st.expander("🔹 Window Functions for Lab Trends"):
            st.code("""
-- Track deterioration of BNP before discharge
SELECT
    l.patient_id,
    l.lab_date,
    l.lab_value AS bnp_value,
    LAG(l.lab_value)  OVER (PARTITION BY l.patient_id ORDER BY l.lab_date) AS prev_bnp,
    LEAD(l.lab_value) OVER (PARTITION BY l.patient_id ORDER BY l.lab_date) AS next_bnp,
    l.lab_value - LAG(l.lab_value) OVER (PARTITION BY l.patient_id ORDER BY l.lab_date) AS bnp_change,
    AVG(l.lab_value) OVER (
        PARTITION BY l.patient_id
        ORDER BY l.lab_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3_avg
FROM labs l
WHERE l.lab_name = 'BNP'
ORDER BY l.patient_id, l.lab_date;
            """, language='sql')

        with st.expander("🔹 Readmission Rate by DRG / Diagnosis Group"):
            st.code("""
-- CMS HRRP uses DRG codes to group diagnoses
-- Simplified: use primary diagnosis category
WITH primary_dx AS (
    SELECT
        encounter_id,
        diagnosis_name,
        ROW_NUMBER() OVER (PARTITION BY encounter_id ORDER BY diagnosis_id) AS dx_rank
    FROM diagnoses
    WHERE is_primary = 1
),
dx_readmit AS (
    SELECT
        d.diagnosis_name,
        COUNT(*) AS total_encounters,
        SUM(e.readmitted_30d) AS readmissions,
        AVG(e.readmitted_30d) * 100 AS readmit_rate_pct,
        AVG(e.length_of_stay) AS avg_los
    FROM encounters e
    JOIN primary_dx d ON e.encounter_id = d.encounter_id
    WHERE d.dx_rank = 1
    GROUP BY d.diagnosis_name
    HAVING COUNT(*) >= 10
)
SELECT *
FROM dx_readmit
ORDER BY readmit_rate_pct DESC;
            """, language='sql')

        with st.expander("🔹 Medicare Spending Per Beneficiary (MSPB) Proxy"):
            st.code("""
-- Approximate total cost of care per patient episode
-- Real MSPB uses Medicare claims — this simulates the concept
WITH episode_costs AS (
    SELECT
        e.patient_id,
        e.encounter_id,
        e.admission_date,
        e.discharge_date,
        e.length_of_stay,
        COUNT(DISTINCT l.lab_id) * 150 AS estimated_lab_cost,
        COUNT(DISTINCT m.med_id) * 25  AS estimated_med_cost,
        e.length_of_stay * 2500        AS estimated_room_cost
    FROM encounters e
    LEFT JOIN labs         l ON e.encounter_id = l.encounter_id
    LEFT JOIN medications  m ON e.encounter_id = m.encounter_id
    GROUP BY e.encounter_id
)
SELECT
    patient_id,
    COUNT(*) AS episodes,
    SUM(estimated_lab_cost + estimated_med_cost + estimated_room_cost) AS total_episode_cost,
    AVG(estimated_lab_cost + estimated_med_cost + estimated_room_cost) AS avg_episode_cost
FROM episode_costs
GROUP BY patient_id
ORDER BY total_episode_cost DESC;
            """, language='sql')

    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">Query History</div>', unsafe_allow_html=True)

        if analytics.query_history:
            hist = pd.DataFrame(analytics.query_history)
            st.dataframe(hist, use_container_width=True)
            idx = st.selectbox("View query",
                               range(len(hist)),
                               format_func=lambda i: f"Query {i+1} — {hist.iloc[i]['timestamp']}")
            st.code(hist.iloc[idx]['query'], language='sql')
            st.info(f"Rows returned: {hist.iloc[idx]['rows_returned']:,}")
        else:
            st.info("No queries in history yet. Run a query to see it here.")
