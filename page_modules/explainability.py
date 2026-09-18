"""
Model Explainability — US Healthcare Edition
SHAP-based global + local explanations with clinical narrative
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def show(train_data, test_data):
    st.markdown("## 🧠 Model Explainability — SHAP Analysis")

    st.markdown("""
    <div class="story-banner">
    <strong>Why explainability matters in healthcare AI:</strong>
    The FDA's AI/ML-based SaMD guidance, CMS requirements for CDS tools, and The Joint Commission's
    standards all require that AI-driven clinical decisions be <em>explainable</em> to clinicians.
    <strong>SHAP (SHapley Additive exPlanations)</strong> is the industry standard — used by Epic,
    Google Health, and Microsoft Healthcare — to explain which features drove each prediction,
    enabling clinicians to validate model reasoning and build trust.
    </div>
    """, unsafe_allow_html=True)

    if not Path('models/best_model.pkl').exists():
        st.warning("⚠️ No trained model found. Train models in the **AI / ML Models** module first.")
        return

    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    st.success("✅ Model loaded for explanation")

    from page_modules.ml_models import prepare_data
    X_train, X_test, y_train, y_test, _ = prepare_data(train_data, test_data)

    tab1, tab2, tab3 = st.tabs([
        "🌍 Global Explanations",
        "🎯 Individual Patient Explanation",
        "📊 Feature Interactions"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Global Feature Importance — SHAP Summary</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        <strong>SHAP Mean Absolute Values</strong> show which features contribute most
        to predictions <em>across all patients</em>. Unlike simple feature importance,
        SHAP values are grounded in game theory — each feature's contribution is fairly
        attributed based on its marginal contribution. This is the same framework used in
        Epic's Deterioration Index explanation panel.
        </div>
        """, unsafe_allow_html=True)

        sample_size = min(200, len(X_test))
        min_shap = min(50, sample_size)
        if min_shap < sample_size:
            compute_size = st.slider("SHAP sample size (larger = more accurate, slower)",
                                      min_shap, sample_size, min(100, sample_size), step=25)
        else:
            compute_size = sample_size
            st.info(f"Using all {sample_size} available test samples for SHAP.")

        if st.button("🔬 Compute SHAP Values", type="primary"):
            try:
                import shap
                X_sample = X_test.sample(n=compute_size, random_state=42)
                y_sample  = y_test.iloc[X_sample.index] if hasattr(y_test, 'iloc') else y_test

                with st.spinner(f"Computing SHAP for {compute_size} patients..."):
                    explainer  = shap.TreeExplainer(model)
                    shap_vals  = explainer.shap_values(X_sample)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]

                st.session_state['shap_values'] = shap_vals
                st.session_state['X_sample']    = X_sample
                st.session_state['y_sample']    = y_sample
                st.success(f"✅ SHAP computed for {compute_size} patients")

            except Exception as e:
                st.error(f"SHAP computation error: {e}")

        if 'shap_values' in st.session_state:
            shap_vals = st.session_state['shap_values']
            X_sample  = st.session_state['X_sample']
            y_sample  = st.session_state['y_sample']

            # Mean |SHAP| bar chart
            mean_shap = np.abs(shap_vals).mean(axis=0)
            shap_df   = pd.DataFrame({'Feature': X_sample.columns, 'SHAP Importance': mean_shap})
            shap_df   = shap_df.sort_values('SHAP Importance', ascending=False).head(20)

            # Clinical label map
            label_map = {
                'last_bnp': 'BNP (pg/mL) — Cardiac Stress',
                'comorbidity_score': 'Comorbidity Score',
                'length_of_stay': 'Length of Stay (days)',
                'last_creatinine': 'Creatinine — Renal Function',
                'prior_admissions_6m': 'Prior Admissions (6mo)',
                'last_sodium': 'Sodium — Fluid Balance',
                'last_hemoglobin': 'Hemoglobin — Anemia',
                'hf_med_count': 'HF Medications',
                'total_meds': 'Total Medications',
                'age': 'Age',
            }
            shap_df['Label'] = shap_df['Feature'].apply(
                lambda f: next((v for k, v in label_map.items() if k in f.lower()), f))

            col1, col2 = st.columns([3, 2])

            with col1:
                fig = px.bar(shap_df, x='SHAP Importance', y='Label', orientation='h',
                             color='SHAP Importance', color_continuous_scale='Oranges',
                             title="Top 20 Features — Mean |SHAP| Value (Global Importance)")
                fig.update_layout(height=540, plot_bgcolor="white",
                                  yaxis=dict(autorange="reversed"),
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Clinical Interpretation of Top SHAP Features**")
                insights = {
                    'bnp': ('🫀', 'BNP reflects myocardial wall stress. Values >1000 pg/mL indicate decompensated HF and strongly predict readmission.'),
                    'comorbidity': ('📋', 'Multiple chronic conditions amplify readmission risk. CKD + HF + Diabetes = "triple whammy syndrome."'),
                    'creatinine': ('🧪', 'Rising creatinine at discharge signals cardiorenal syndrome — worsening renal perfusion from low cardiac output.'),
                    'sodium': ('⚗️', 'Hyponatremia (<135 mEq/L) reflects neurohormonal activation in HF and carries 2× readmission risk.'),
                    'stay': ('🏥', 'Prolonged hospitalization correlates with severity of illness and discharge to a higher level of care.'),
                    'prior': ('🔄', 'Prior admissions are the strongest behavioral predictor — patients with a pattern of decompensation continue the cycle.'),
                    'hemoglobin': ('🩸', 'Anemia reduces oxygen delivery to a failing heart, worsening symptoms and increasing early readmission.'),
                }
                for _, row in shap_df.head(8).iterrows():
                    feat = row['Feature'].lower()
                    ic, txt = next(((ic, t) for k, (ic, t) in insights.items() if k in feat),
                                   ('📊', 'Contributing feature identified by SHAP analysis.'))
                    st.markdown(f"""
                    <div style="background:#f8f9fa; border-left:3px solid #f39c12;
                                padding:0.4rem 0.7rem; margin:0.3rem 0;
                                border-radius:0 4px 4px 0; font-size:0.8rem; line-height:1.4;">
                    {ic} <strong>{row['Label']}</strong><br>
                    <span style='color:#555'>{txt}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # SHAP beeswarm (via matplotlib)
            st.markdown('<div class="section-title">SHAP Beeswarm Summary Plot</div>',
                        unsafe_allow_html=True)
            try:
                import shap
                top_feats = shap_df.head(15)['Feature'].tolist()
                idx_top   = [list(X_sample.columns).index(f) for f in top_feats if f in X_sample.columns]
                X_top     = X_sample.iloc[:, idx_top]
                sv_top    = shap_vals[:, idx_top]

                fig_bp, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(sv_top, X_top, feature_names=top_feats, show=False, plot_size=None)
                plt.tight_layout()
                st.pyplot(fig_bp, use_container_width=True)
                plt.close()
            except Exception as e:
                st.warning(f"Beeswarm plot unavailable: {e}")

        else:
            st.info("Click **Compute SHAP Values** above to generate explanations.")

    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Individual Patient Explanation</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        <strong>Local explanations</strong> show <em>why</em> the model predicted a specific
        risk score for one patient — which features pushed the prediction up (red) or down (blue)
        from the population average. This mirrors how Epic Deterioration Index explains scores
        to bedside nurses.
        </div>
        """, unsafe_allow_html=True)

        if 'shap_values' not in st.session_state:
            st.warning("Compute SHAP values in the **Global Explanations** tab first.")
        else:
            shap_vals = st.session_state['shap_values']
            X_sample  = st.session_state['X_sample']
            y_sample  = st.session_state['y_sample']

            proba_all  = model.predict_proba(X_sample)[:, 1]
            X_sample_r = X_sample.reset_index(drop=True)

            patient_idx = st.slider("Select Patient Index",
                                     0, len(X_sample_r)-1, 0,
                                     help="Browse through the sample patients")

            proba_pt  = proba_all[patient_idx]
            risk_pct  = proba_pt * 100
            actual    = int(y_sample.iloc[patient_idx]) if hasattr(y_sample, 'iloc') else 0

            if risk_pct < 15:
                tier, color = "🟢 Low", "#27ae60"
            elif risk_pct < 30:
                tier, color = "🟡 Moderate", "#f39c12"
            elif risk_pct < 50:
                tier, color = "🟠 High", "#e67e22"
            else:
                tier, color = "🔴 Very High", "#e74c3c"

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Risk",   f"{risk_pct:.1f}%")
            c2.metric("Risk Tier",         tier)
            c3.metric("Actual Outcome",
                      "Readmitted ⚠️" if actual == 1 else "Not Readmitted ✅",
                      delta=None)

            # SHAP waterfall via plotly
            sv_pt     = shap_vals[patient_idx]
            feat_vals = X_sample_r.iloc[patient_idx]
            baseline  = shap_vals.mean(axis=0).sum()

            # Top contributors
            contrib_df = pd.DataFrame({
                'Feature': X_sample.columns,
                'SHAP Value': sv_pt,
                'Feature Value': feat_vals.values
            }).sort_values('SHAP Value', key=abs, ascending=False).head(15)

            label_map = {
                'last_bnp': 'BNP', 'comorbidity_score': 'Comorbidities',
                'length_of_stay': 'LOS', 'last_creatinine': 'Creatinine',
                'prior_admissions_6m': 'Prior Admits', 'last_sodium': 'Sodium',
                'last_hemoglobin': 'Hemoglobin', 'hf_med_count': 'HF Meds',
                'total_meds': 'Total Meds', 'age': 'Age',
            }
            contrib_df['Label'] = contrib_df['Feature'].apply(
                lambda f: next((v for k, v in label_map.items() if k in f.lower()), f))
            contrib_df['Label_Val'] = (contrib_df['Label'] + ' = ' +
                                        contrib_df['Feature Value'].round(2).astype(str))

            fig = go.Figure(go.Waterfall(
                orientation='h',
                measure=['relative'] * len(contrib_df),
                y=contrib_df['Label_Val'],
                x=contrib_df['SHAP Value'],
                connector={"line": {"color": "gray", "width": 0.5}},
                decreasing={"marker": {"color": "#3498db"}},
                increasing={"marker": {"color": "#e74c3c"}}
            ))
            fig.add_vline(x=0, line_color='black', line_width=1)
            fig.update_layout(
                title=f"Patient #{patient_idx} — SHAP Waterfall Explanation<br><sub>Red = pushes risk UP | Blue = pushes risk DOWN</sub>",
                xaxis_title="SHAP Value (contribution to prediction)",
                height=500, plot_bgcolor="white",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)

            # Patient data summary
            with st.expander("📋 View Patient Clinical Data"):
                pt_data = X_sample_r.iloc[[patient_idx]].T.reset_index()
                pt_data.columns = ['Feature', 'Value']
                pt_data = pt_data[pt_data['Value'] != 0].head(30)
                st.dataframe(pt_data, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Feature Interaction Analysis</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        <strong>SHAP Dependence Plots</strong> show how a feature's contribution to
        readmission risk changes across its value range, and how a second feature modulates
        that effect. These interactions reveal clinically meaningful patterns —
        e.g., elevated BNP matters even more when creatinine is also high (cardiorenal syndrome).
        </div>
        """, unsafe_allow_html=True)

        if 'shap_values' not in st.session_state:
            st.warning("Compute SHAP values in the **Global Explanations** tab first.")
        else:
            shap_vals = st.session_state['shap_values']
            X_sample  = st.session_state['X_sample']

            top_feats = (pd.DataFrame({'f': X_sample.columns,
                                       's': np.abs(shap_vals).mean(axis=0)})
                         .sort_values('s', ascending=False).head(10)['f'].tolist())

            col_main, col_color = st.columns(2)
            with col_main:
                main_feat  = st.selectbox("Main Feature",  top_feats, index=0)
            with col_color:
                color_feat = st.selectbox("Color by (interaction)", top_feats, index=min(1, len(top_feats)-1))

            if main_feat in X_sample.columns and color_feat in X_sample.columns:
                main_idx  = list(X_sample.columns).index(main_feat)
                shap_main = shap_vals[:, main_idx]
                feat_vals = X_sample[main_feat].values
                color_vals= X_sample[color_feat].values

                fig = go.Figure(go.Scatter(
                    x=feat_vals, y=shap_main,
                    mode='markers',
                    marker=dict(color=color_vals, colorscale='RdYlBu_r',
                                size=7, opacity=0.7,
                                colorbar=dict(title=color_feat)),
                    text=[f"{main_feat}={fv:.2f}<br>{color_feat}={cv:.2f}<br>SHAP={sv:.4f}"
                          for fv, cv, sv in zip(feat_vals, color_vals, shap_main)],
                    hovertemplate='%{text}<extra></extra>'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title=f"SHAP Dependence: {main_feat} (colored by {color_feat})",
                    xaxis_title=main_feat, yaxis_title=f"SHAP Value for {main_feat}",
                    height=480, plot_bgcolor="white"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Correlation table of top SHAP interactions
                st.markdown('<div class="section-title">SHAP × Feature Correlation Matrix (Top 8)</div>',
                            unsafe_allow_html=True)
                top8 = top_feats[:8]
                idx8 = [list(X_sample.columns).index(f) for f in top8]
                sv8  = shap_vals[:, idx8]
                corr8= np.corrcoef(sv8.T)
                fig2 = go.Figure(go.Heatmap(
                    z=corr8, x=top8, y=top8,
                    colorscale='RdBu', zmid=0,
                    text=np.round(corr8, 2), texttemplate='%{text}',
                    textfont={"size": 9}
                ))
                fig2.update_layout(title="Correlation of SHAP Values (Feature Interaction Proxy)",
                                   height=420)
                st.plotly_chart(fig2, use_container_width=True)
