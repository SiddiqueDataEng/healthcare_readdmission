"""
Patient Risk Predictions — US Healthcare Edition
Single patient + batch predictions with clinical context and care plans
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def show(train_data, test_data):
    st.markdown("## 🎯 Patient Readmission Risk Predictions")

    st.markdown("""
    <div class="story-banner">
    <strong>Clinical Decision Support:</strong> This module functions as a
    <em>Clinical Decision Support (CDS) tool</em> — generating individualized 30-day readmission
    risk scores to guide care transition planning. Risk scores are presented alongside
    <strong>evidence-based intervention recommendations</strong> aligned with the
    Society of Hospital Medicine's Project BOOST and CMS Transitional Care Management guidelines.
    </div>
    """, unsafe_allow_html=True)

    from page_modules.ml_models import get_prediction_model
    try:
        model, feature_names = get_prediction_model(train_data, test_data)
    except Exception as exc:
        st.error(f"Unable to prepare a compatible prediction model: {type(exc).__name__}")
        return

    st.success("✅ Predictive model loaded and ready")

    tab1, tab2, tab3 = st.tabs([
        "👤 Single Patient Assessment",
        "📊 Batch Risk Scoring",
        "📋 Risk Score Interpretation"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Individual Patient Risk Assessment</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        Enter patient demographics and clinical values at discharge. The AI model generates a
        <strong>probability score (0–100%)</strong> for 30-day readmission and a
        <strong>risk tier</strong> with corresponding care transition recommendations.
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Demographics**")
            age = st.number_input("Age (years)", 18, 120, 72)
            gender = st.selectbox("Gender", ["M", "F"])
            race = st.selectbox("Race/Ethnicity",
                                ["White", "Black", "Hispanic", "Asian", "Other"])
            insurance_type = st.selectbox("Insurance Type",
                                          ["Medicare", "Medicaid", "Private", "Self-pay"])

        with col2:
            st.markdown("**🧪 Discharge Lab Values**")
            last_bnp        = st.number_input("BNP (pg/mL)",        0.0, 10000.0, 650.0, step=50.0)
            last_sodium     = st.number_input("Sodium (mEq/L)",     100.0, 160.0, 138.0)
            last_creatinine = st.number_input("Creatinine (mg/dL)", 0.0,  15.0,   1.4)
            last_hemoglobin = st.number_input("Hemoglobin (g/dL)",  4.0,  20.0,   11.5)

        with col3:
            st.markdown("**🏥 Hospital & Discharge**")
            length_of_stay        = st.number_input("Length of Stay (days)", 1, 60, 6)
            discharge_location    = st.selectbox("Discharge Location",    ["Home", "Facility", "SNF", "Rehab"])
            discharge_disposition = st.selectbox("Discharge Disposition", ["Home", "SNF", "Hospice", "AMA", "Rehab"])
            weekend_discharge     = st.checkbox("Weekend Discharge", value=False)
            prior_admissions_6m   = st.number_input("Prior Admissions (6 months)", 0, 20, 1)
            prior_ed_visits_30d   = st.number_input("ED Visits (30 days prior)",   0, 10, 0)

        st.markdown("**📋 Comorbidities**")
        cmorb1, cmorb2, cmorb3, cmorb4 = st.columns(4)
        with cmorb1:
            has_hypertension = st.checkbox("Hypertension")
            has_diabetes     = st.checkbox("Diabetes")
        with cmorb2:
            has_cad  = st.checkbox("Coronary Artery Disease")
            has_copd = st.checkbox("COPD")
        with cmorb3:
            has_ckd      = st.checkbox("Chronic Kidney Disease")
            has_afib     = st.checkbox("Atrial Fibrillation")
        with cmorb4:
            has_obesity = st.checkbox("Obesity")
            has_anemia  = st.checkbox("Anemia")

        comorbidity_score = sum([has_hypertension, has_diabetes, has_cad, has_copd,
                                  has_ckd, has_afib, has_obesity, has_anemia])

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            hf_med_count = st.number_input("HF Medications Count", 0, 15, 3)
        with col_m2:
            total_meds = st.number_input("Total Medications at Discharge", 0, 30, 8)

        if st.button("🔮 Calculate Readmission Risk", type="primary", use_container_width=True):
            # Build patient row
            patient_dict = {
                'age': age, 'length_of_stay': length_of_stay,
                'last_bnp': last_bnp, 'last_sodium': last_sodium,
                'last_creatinine': last_creatinine, 'last_hemoglobin': last_hemoglobin,
                'comorbidity_score': comorbidity_score, 'hf_med_count': hf_med_count,
                'total_meds': total_meds, 'prior_admissions_6m': prior_admissions_6m,
                'prior_ed_visits_30d': prior_ed_visits_30d,
                'weekend_discharge': int(weekend_discharge),
                'has_hypertension': int(has_hypertension), 'has_diabetes': int(has_diabetes),
                'has_cad': int(has_cad), 'has_copd': int(has_copd),
                'has_ckd': int(has_ckd), 'has_afib': int(has_afib),
                'has_obesity': int(has_obesity), 'has_anemia': int(has_anemia),
                f'gender_{gender}': 1, f'race_{race}': 1,
                f'insurance_type_{insurance_type}': 1,
                f'discharge_location_{discharge_location}': 1,
                f'discharge_disposition_{discharge_disposition}': 1,
            }

            patient_df = pd.DataFrame([patient_dict])
            for col in feature_names:
                if col not in patient_df.columns:
                    patient_df[col] = 0
            patient_df = patient_df[feature_names].fillna(0).astype('float64')

            proba = model.predict_proba(patient_df)[0][1]
            risk_pct = proba * 100

            if risk_pct < 15:
                tier, color, icon = "🟢 Low Risk", "#27ae60", "✅"
            elif risk_pct < 30:
                tier, color, icon = "🟡 Moderate Risk", "#f39c12", "⚠️"
            elif risk_pct < 50:
                tier, color, icon = "🟠 High Risk", "#e67e22", "🔴"
            else:
                tier, color, icon = "🔴 Very High Risk", "#e74c3c", "🚨"

            st.markdown("---")
            c1, c2, c3 = st.columns([2, 2, 3])

            with c1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_pct,
                    title={"text": "30-Day Readmission Risk (%)"},
                    number={"suffix": "%", "font": {"size": 36, "color": color}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": color},
                        "steps": [
                            {"range": [0, 15],  "color": "#d5f5e3"},
                            {"range": [15, 30], "color": "#fef9e7"},
                            {"range": [30, 50], "color": "#fdebd0"},
                            {"range": [50, 100],"color": "#fadbd8"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 3}, "value": 22}
                    }
                ))
                fig.update_layout(height=280)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown(f"""
                <div style="background:{color}15; border: 2px solid {color};
                            border-radius:12px; padding:1.5rem; text-align:center; margin-top:1rem;">
                    <div style="font-size:2.5rem;">{icon}</div>
                    <div style="font-size:1.5rem; font-weight:700; color:{color};">{tier}</div>
                    <div style="font-size:2.2rem; font-weight:800; color:{color};">{risk_pct:.1f}%</div>
                    <div style="font-size:0.85rem; color:#555;">30-Day Readmission Probability</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                # Key risk factors for this patient
                st.markdown("**📊 Patient Risk Factor Summary**")
                factors = []
                if last_bnp > 1000:    factors.append(("🫀 BNP severely elevated",         "high"))
                if last_creatinine > 1.5: factors.append(("🧪 Creatinine elevated (CKD risk)", "high"))
                if last_sodium < 135:  factors.append(("⚗️ Hyponatremia present",            "high"))
                if last_hemoglobin < 10: factors.append(("🩸 Significant anemia",            "high"))
                if comorbidity_score >= 4: factors.append(("📋 High comorbidity burden",     "high"))
                if length_of_stay > 7:  factors.append(("🏥 Prolonged LOS (>7 days)",       "mod"))
                if prior_admissions_6m > 0: factors.append(("🔄 Prior admission in 6 months","high"))
                if weekend_discharge:  factors.append(("📅 Weekend discharge",               "mod"))
                if discharge_disposition in ["SNF", "Hospice"]: factors.append(("🏠 High-acuity discharge setting", "mod"))
                if age > 80:           factors.append(("👤 Age >80 (frailty risk)",          "mod"))

                if factors:
                    for fact, level in factors[:6]:
                        bg = "#fff0f0" if level == "high" else "#fffbea"
                        bc = "#e74c3c" if level == "high" else "#f39c12"
                        st.markdown(f"""
                        <div style="background:{bg}; border-left:3px solid {bc};
                                    padding:0.35rem 0.7rem; margin:0.25rem 0;
                                    border-radius:0 4px 4px 0; font-size:0.85rem;">{fact}</div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-info">✅ No major risk flags identified at this threshold</div>
                    """, unsafe_allow_html=True)

            # Care plan recommendations
            st.markdown("---")
            st.markdown('<div class="section-title">📋 Recommended Care Transition Plan</div>',
                        unsafe_allow_html=True)

            care_plans = {
                "🟢 Low Risk": [
                    ("📋 Standard discharge instructions (written + verbal)", "Before discharge"),
                    ("📞 Courtesy call at 7 days", "Day 7"),
                    ("🏥 PCP follow-up scheduled within 14 days", "Within 14 days"),
                ],
                "🟡 Moderate Risk": [
                    ("📋 Enhanced discharge education with teach-back confirmation", "Before discharge"),
                    ("💊 Medication reconciliation & patient education", "Before discharge"),
                    ("📞 Nurse phone call within 48 hours of discharge", "Day 1–2"),
                    ("🏥 PCP follow-up within 7 days — confirmed appointment", "Within 7 days"),
                    ("📊 Weight monitoring instruction (daily weigh-in)", "Before discharge"),
                ],
                "🟠 High Risk": [
                    ("👩‍⚕️ Assign dedicated care transition coach", "Before discharge"),
                    ("💊 Pharmacist medication reconciliation review", "Before discharge"),
                    ("📞 Care coordinator call within 24 hours", "Day 1"),
                    ("🏥 Cardiology follow-up within 3–5 days", "Within 5 days"),
                    ("📱 Remote monitoring enrollment (weight, BP, symptoms)", "Day 1–3"),
                    ("🚗 Transportation assistance arranged", "Before discharge"),
                ],
                "🔴 Very High Risk": [
                    ("🚨 Intensive Care Transition Program enrollment", "Before discharge"),
                    ("👩‍⚕️ Case manager + social work assessment", "Before discharge"),
                    ("🏠 Home health nursing visit — 3×/week × 2 weeks", "Day 1–14"),
                    ("📱 Daily remote monitoring check-in (weight + symptoms)", "Daily"),
                    ("🏥 Cardiology/HF clinic visit within 48–72 hours", "Day 2–3"),
                    ("💊 Pharmacist home visit or tele-medication review", "Day 1–3"),
                    ("🍽️ Dietary consult — low-sodium diet reinforcement", "Before discharge"),
                    ("📋 Palliative care consult if appropriate", "As indicated"),
                ],
            }

            actions = care_plans.get(tier, care_plans["🟢 Low Risk"])
            plan_df = pd.DataFrame(actions, columns=["Intervention", "Timing"])
            plan_df.insert(0, "#", range(1, len(plan_df)+1))
            st.dataframe(plan_df, use_container_width=True, hide_index=True)

            # Export care plan
            st.download_button(
                "📥 Download Care Transition Plan",
                plan_df.to_csv(index=False),
                file_name=f"care_plan_risk_{risk_pct:.0f}pct.csv",
                mime="text/csv"
            )

    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Batch Risk Scoring — Entire Patient Cohort</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        Score your entire discharge cohort at once. Upload a CSV or use the test dataset.
        This workflow mirrors the <strong>Epic Discharge Risk Worklist</strong> and
        <strong>Cerner Readmission Navigator</strong> used in US health systems.
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload patient CSV (optional — uses test dataset if not uploaded)",
                                    type=['csv'])

        if uploaded:
            batch_df = pd.read_csv(uploaded)
            st.success(f"✅ Uploaded {len(batch_df):,} patients")
        else:
            batch_df = test_data.copy()
            st.info(f"Using test dataset: {len(batch_df):,} patients")

        if st.button("🔮 Score All Patients", type="primary"):
            try:
                from page_modules.ml_models import prepare_data
                X_tr, X_te, y_tr, y_te, _ = prepare_data(train_data, batch_df)

                proba_all = model.predict_proba(X_te)[:, 1]
                results_df = batch_df.copy().reset_index(drop=True)
                results_df['readmission_risk_pct'] = (proba_all * 100).round(1)
                results_df['risk_tier'] = pd.cut(
                    results_df['readmission_risk_pct'],
                    bins=[-1, 15, 30, 50, 101],
                    labels=['🟢 Low', '🟡 Moderate', '🟠 High', '🔴 Very High']
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Patients Scored", f"{len(results_df):,}")
                c2.metric("High + Very High Risk",
                          f"{(results_df['risk_tier'].isin(['🟠 High', '🔴 Very High'])).sum():,}")
                c3.metric("Avg Predicted Risk",
                          f"{results_df['readmission_risk_pct'].mean():.1f}%")
                c4.metric("Actual Readmit Rate",
                          f"{batch_df['readmitted_30d'].mean()*100:.1f}%" if 'readmitted_30d' in batch_df.columns else "N/A")

                col_v, col_t = st.columns([2, 3])

                with col_v:
                    tier_counts = results_df['risk_tier'].value_counts().reset_index()
                    tier_counts.columns = ['tier', 'count']
                    fig = px.pie(tier_counts, values='count', names='tier',
                                 title="Risk Tier Distribution",
                                 color_discrete_sequence=['#27ae60','#f39c12','#e67e22','#e74c3c'],
                                 hole=0.45)
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

                with col_t:
                    # Risk score distribution
                    fig = px.histogram(results_df, x='readmission_risk_pct', nbins=40,
                                       title="Distribution of Predicted Risk Scores",
                                       color_discrete_sequence=['#1a6eb5'],
                                       labels={'readmission_risk_pct': 'Predicted 30-Day Risk (%)'})
                    fig.add_vline(x=22, line_dash="dash", annotation_text="National Avg 22%",
                                  line_color="red")
                    fig.update_layout(height=350, plot_bgcolor="white")
                    st.plotly_chart(fig, use_container_width=True)

                # High risk patient list
                st.markdown('<div class="section-title">🚨 High-Risk Patient Worklist</div>',
                            unsafe_allow_html=True)

                high_risk_list = results_df[results_df['risk_tier'].isin(['🟠 High', '🔴 Very High'])].sort_values(
                    'readmission_risk_pct', ascending=False)

                disp_cols = [c for c in ['patient_id', 'age', 'gender', 'discharge_location',
                                          'length_of_stay', 'comorbidity_score', 'last_bnp',
                                          'readmission_risk_pct', 'risk_tier', 'readmitted_30d']
                             if c in high_risk_list.columns]
                disp = high_risk_list[disp_cols].head(50)

                if 'patient_id' in disp.columns:
                    disp = disp.copy()
                    disp['patient_id'] = disp['patient_id'].astype(str).str[:8] + "****"

                st.dataframe(
                    disp.style.map(
                        lambda v: 'background-color:#fadbd8; font-weight:bold' if v == '🔴 Very High'
                                  else ('background-color:#fdebd0' if v == '🟠 High' else ''),
                        subset=['risk_tier'] if 'risk_tier' in disp.columns else []
                    ),
                    use_container_width=True, height=400
                )

                st.download_button(
                    "📥 Download Full Risk-Scored Patient List",
                    results_df.to_csv(index=False),
                    file_name="batch_risk_scores.csv", mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Batch scoring error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Risk Score Interpretation Guide</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        This guide explains how to interpret and act on readmission risk scores —
        designed for nurses, case managers, hospitalists, and care coordinators.
        </div>
        """, unsafe_allow_html=True)

        tiers = [
            ("🟢 Low Risk", "< 15%", "#27ae60", "#d5f5e3",
             "Standard discharge. PCP follow-up within 14 days. Written instructions.",
             "Call at 7 days if concerns; otherwise routine follow-up."),
            ("🟡 Moderate Risk", "15–30%", "#f39c12", "#fef9e7",
             "Enhanced discharge education. Medication reconciliation. 48h nurse call. PCP within 7 days.",
             "Confirm appointment. Assess barriers to follow-up. Remote weight monitoring."),
            ("🟠 High Risk", "30–50%", "#e67e22", "#fdebd0",
             "Care transition coach. Pharmacist review. Cardiology within 5 days. Remote monitoring enrollment.",
             "Daily weight log. Symptoms checklist. Transportation arranged. Escalation path defined."),
            ("🔴 Very High Risk", "> 50%", "#e74c3c", "#fadbd8",
             "Intensive Care Transition Program. Home health. HF clinic within 48–72h. Daily monitoring.",
             "Social work involved. Housing/food security assessed. Palliative care if appropriate."),
        ]

        for tier_name, pct, color, bg, initial, followup in tiers:
            st.markdown(f"""
            <div style="background:{bg}; border:2px solid {color}; border-radius:10px;
                        padding:1rem 1.3rem; margin:0.7rem 0;">
                <div style="font-size:1.1rem; font-weight:700; color:{color}; margin-bottom:0.4rem;">
                    {tier_name} — {pct}
                </div>
                <div style="font-size:0.88rem; color:#333; margin-bottom:0.3rem;">
                    <strong>Discharge Actions:</strong> {initial}
                </div>
                <div style="font-size:0.88rem; color:#555;">
                    <strong>Post-Discharge Follow-up:</strong> {followup}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-info">
        <strong>Important Disclaimer:</strong> This risk score is a <em>clinical decision support tool</em>,
        not a clinical decision maker. Final care decisions must be made by qualified clinicians
        in the context of the full clinical picture. This tool has not been validated as FDA-cleared
        SaMD. Use in conjunction with clinical judgment.
        </div>
        """, unsafe_allow_html=True)
