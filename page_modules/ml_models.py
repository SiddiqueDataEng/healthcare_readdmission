"""
AI / ML Models — US Healthcare Edition
Clinical context, model comparison, calibration, fairness analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             auc, precision_recall_curve, average_precision_score,
                             brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import pickle
from pathlib import Path


# ── Data preparation ──────────────────────────────────────────────────────────
def prepare_data(train_data, test_data):
    exclude = ['readmitted_30d', 'index_encounter_id', 'patient_id',
               'index_admission', 'index_discharge', 'readmission_date',
               'days_to_readmit', 'died_within_30d']
    feature_cols = [c for c in train_data.columns if c not in exclude]
    cat_cols = train_data[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

    X_train = pd.get_dummies(train_data[feature_cols].copy(), columns=cat_cols, drop_first=True)
    X_test  = pd.get_dummies(test_data[feature_cols].copy(),  columns=cat_cols, drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    def clean(col):
        for ch in ['[',']','<','>',',','(',')']:
            col = col.replace(ch, '_')
        while '__' in col: col = col.replace('__', '_')
        return col.strip('_')

    X_train.columns = [clean(c) for c in X_train.columns]
    X_test.columns  = [clean(c) for c in X_test.columns]

    for col in X_train.columns:
        med = X_train[col].median() if X_train[col].dtype in ['float64','int64'] else 0
        med = med if not pd.isna(med) else 0
        X_train[col] = X_train[col].fillna(med)
        X_test[col]  = X_test[col].fillna(med)

    X_train = X_train.fillna(0).astype('float64')
    X_test  = X_test.fillna(0).astype('float64')

    y_train = train_data['readmitted_30d']
    y_test  = test_data['readmitted_30d']
    return X_train, X_test, y_train, y_test, feature_cols


@st.cache_resource
def train_models(X_train, y_train, use_smote=True, _version=3):
    if use_smote:
        class_counts = pd.Series(y_train).value_counts()
        minority_count = int(class_counts.min()) if len(class_counts) > 1 else 0
        if minority_count > 1:
            sm = SMOTE(random_state=42, k_neighbors=min(5, minority_count - 1))
            X_r, y_r = sm.fit_resample(X_train, y_train)
        else:
            X_r, y_r = X_train, y_train
    else:
        X_r, y_r = X_train, y_train

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=0.5),
        'Random Forest':       RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced'),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'XGBoost':             xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss', verbosity=0),
        'LightGBM':            lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1),
    }
    trained = {}
    for name, mdl in models.items():
        mdl.fit(X_r, y_r)
        trained[name] = mdl
    return trained


def get_prediction_model(train_data, test_data):
    """Return a session, artifact, or freshly trained prediction model."""
    model = st.session_state.get('prediction_model')
    feature_names = st.session_state.get('model_feature_names')
    if model is not None and feature_names is not None:
        return model, feature_names

    model_path = Path('models/best_model.pkl')
    features_path = Path('models/feature_names.pkl')
    if model_path.exists() and features_path.exists():
        try:
            with model_path.open('rb') as handle:
                model = pickle.load(handle)
            with features_path.open('rb') as handle:
                feature_names = pickle.load(handle)
            return model, feature_names
        except (ModuleNotFoundError, AttributeError, ValueError, pickle.UnpicklingError):
            pass

    X_train, _, y_train, _, _ = prepare_data(train_data, test_data)
    trained_models = train_models(X_train, y_train, True, _version=4)
    model = trained_models.get('LightGBM', next(iter(trained_models.values())))
    feature_names = X_train.columns.tolist()
    st.session_state['trained_models'] = trained_models
    st.session_state['model_feature_names'] = feature_names
    st.session_state['prediction_model'] = model
    return model, feature_names


def evaluate_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm      = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    report  = classification_report(
        y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0
    )
    brier   = brier_score_loss(y_test, y_proba)
    tn, fp, fn, tp = cm.ravel()
    return {
        'predictions': y_pred, 'probabilities': y_proba,
        'confusion_matrix': cm, 'fpr': fpr, 'tpr': tpr,
        'roc_auc': auc(fpr, tpr),
        'precision': prec, 'recall': rec,
        'avg_precision': average_precision_score(y_test, y_proba),
        'report': report, 'brier': brier,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


# ─────────────────────────────────────────────────────────────────────────────
def show(train_data, test_data):
    st.markdown("## 🤖 AI / Machine Learning Models")

    st.markdown("""
    <div class="story-banner">
    <strong>Clinical AI Context:</strong> Predictive models for 30-day readmission are deployed
    at hundreds of US health systems (Epic Deterioration Index, Cerner HealtheIntent, Health Catalyst).
    The FDA's SaMD (Software as a Medical Device) framework governs clinical decision support AI.
    Models must be evaluated not just on accuracy, but on <strong>calibration</strong> (do predicted
    probabilities match actual rates?), <strong>fairness</strong> (does the model perform equally
    across demographic groups?), and <strong>clinical utility</strong> (does acting on predictions
    improve outcomes?).
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Preparing features..."):
        X_train, X_test, y_train, y_test, feature_cols = prepare_data(train_data, test_data)

    c1, c2, c3 = st.columns(3)
    c1.metric("Training Samples", f"{len(X_train):,}")
    c2.metric("Test Samples",     f"{len(X_test):,}")
    c3.metric("Features",         f"{X_train.shape[1]:,}")

    st.markdown("---")

    col_cfg, col_btn = st.columns([2,1])
    with col_cfg:
        use_smote = st.checkbox("Apply SMOTE (balance class imbalance)", value=True,
                                help="SMOTE synthesizes minority class samples. Recommended for imbalanced HF readmission data (~1.3% positive rate).")
    with col_btn:
        train_btn = st.button("🚀 Train All Models", type="primary", use_container_width=True)
        if st.button("🔄 Reset Cache", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    if train_btn:
        st.session_state['models_trained'] = True

    if 'models_trained' not in st.session_state:
        st.info("Click **Train All Models** to begin. Training 5 models typically takes 15–60 seconds.")
        return

    with st.spinner("Training models..."):
        trained_models = train_models(X_train, y_train, use_smote, _version=3)
    st.session_state['trained_models'] = trained_models
    st.session_state['model_feature_names'] = X_train.columns.tolist()
    st.session_state['prediction_model'] = trained_models.get(
        'LightGBM', next(iter(trained_models.values())))
    st.success("✅ All 5 models trained successfully!")

    # Evaluate all
    results = {name: evaluate_model(mdl, X_test, y_test) for name, mdl in trained_models.items()}

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Model Comparison",
        "🎯 Detailed Metrics",
        "📈 ROC / PR Curves",
        "📊 Calibration & Threshold",
        "🔍 Feature Importance"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Model Performance Leaderboard</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="story-banner">
        <strong>ROC-AUC</strong> measures overall discrimination (0.5 = random, 1.0 = perfect).
        For readmission prediction, ROC-AUC of 0.70–0.85 is typical in the literature.
        <strong>Precision</strong> = of patients flagged as high-risk, how many truly were?
        <strong>Recall (Sensitivity)</strong> = of all actual readmissions, how many did we catch?
        In healthcare, <strong>high recall</strong> is usually prioritized to avoid missing at-risk patients.
        </div>
        """, unsafe_allow_html=True)

        comp = []
        for name, r in results.items():
            rep = r['report']
            tn, fp, fn, tp = r['tn'], r['fp'], r['fn'], r['tp']
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            comp.append({
                'Model': name,
                'ROC-AUC': r['roc_auc'],
                'Avg Precision': r['avg_precision'],
                'Accuracy': rep['accuracy'],
                'Precision': rep['1']['precision'],
                'Recall (Sens.)': rep['1']['recall'],
                'F1-Score': rep['1']['f1-score'],
                'Specificity': spec,
                'Brier Score': r['brier']
            })
        comp_df = pd.DataFrame(comp).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
        comp_df.index = comp_df.index + 1
        comp_df.insert(0, 'Rank', comp_df.index)

        st.dataframe(
            comp_df.style.format({
                'ROC-AUC': '{:.4f}', 'Avg Precision': '{:.4f}', 'Accuracy': '{:.4f}',
                'Precision': '{:.4f}', 'Recall (Sens.)': '{:.4f}', 'F1-Score': '{:.4f}',
                'Specificity': '{:.4f}', 'Brier Score': '{:.4f}'
            }).background_gradient(cmap='RdYlGn', subset=['ROC-AUC', 'Recall (Sens.)'])
              .background_gradient(cmap='RdYlGn_r', subset=['Brier Score']),
            use_container_width=True, hide_index=True
        )

        best = comp_df.iloc[0]['Model']
        best_auc = comp_df.iloc[0]['ROC-AUC']
        st.success(f"🏆 Best Model: **{best}** — ROC-AUC: {best_auc:.4f}")

        # Radar chart comparison
        metrics = ['ROC-AUC', 'Precision', 'Recall (Sens.)', 'F1-Score', 'Specificity']
        colors  = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']
        fig = go.Figure()
        for i, row in comp_df.iterrows():
            vals = [row[m] for m in metrics]
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=metrics + [metrics[0]],
                fill='toself', name=row['Model'],
                line_color=colors[(i-1) % len(colors)]
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Model Performance Radar", height=450, showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Detailed Model Analysis</div>', unsafe_allow_html=True)

        sel = st.selectbox("Select Model", list(trained_models.keys()))
        r   = results[sel]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC",    f"{r['roc_auc']:.4f}")
        c2.metric("Recall",     f"{r['report']['1']['recall']:.4f}",
                  help="True Positive Rate — critical in healthcare to minimize missed readmissions")
        c3.metric("Precision",  f"{r['report']['1']['precision']:.4f}")
        c4.metric("Brier Score",f"{r['brier']:.4f}",
                  help="Lower = better calibrated predictions. 0 = perfect, 0.25 = random")

        col1, col2 = st.columns(2)

        with col1:
            cm = r['confusion_matrix']
            tn, fp, fn, tp = r['tn'], r['fp'], r['fn'], r['tp']
            total = tn + fp + fn + tp
            cm_labels = [
                [f"TN: {tn}<br>({tn/total*100:.1f}%)<br>Correctly Not Flagged",
                 f"FP: {fp}<br>({fp/total*100:.1f}%)<br>Unnecessary Intervention"],
                [f"FN: {fn}<br>({fn/total*100:.1f}%)<br>Missed Readmission ⚠️",
                 f"TP: {tp}<br>({tp/total*100:.1f}%)<br>Correctly Flagged ✅"]
            ]
            fig = go.Figure(go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=['Predicted: No Readmit', 'Predicted: Readmit'],
                y=['Actual: No Readmit', 'Actual: Readmit'],
                text=cm_labels, texttemplate="%{text}",
                textfont={"size": 11},
                colorscale='Blues'
            ))
            fig.update_layout(title=f"Confusion Matrix — {sel}", height=380)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="alert-warning">
            <strong>Clinical Note:</strong> False Negatives (missed readmissions) carry higher clinical
            risk than False Positives (unnecessary outreach calls). Threshold adjustment can
            shift this tradeoff.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            rep  = r['report']
            rdf  = pd.DataFrame([
                {'Class': 'Not Readmitted (0)', 'Precision': rep['0']['precision'],
                 'Recall': rep['0']['recall'], 'F1-Score': rep['0']['f1-score'], 'Support': rep['0']['support']},
                {'Class': 'Readmitted (1)', 'Precision': rep['1']['precision'],
                 'Recall': rep['1']['recall'], 'F1-Score': rep['1']['f1-score'], 'Support': rep['1']['support']},
                {'Class': 'Macro Avg', 'Precision': rep['macro avg']['precision'],
                 'Recall': rep['macro avg']['recall'], 'F1-Score': rep['macro avg']['f1-score'],
                 'Support': rep['macro avg']['support']},
            ])
            st.markdown("**Classification Report**")
            st.dataframe(
                rdf.style.format({'Precision':'{:.4f}','Recall':'{:.4f}','F1-Score':'{:.4f}','Support':'{:.0f}'})
                   .background_gradient(cmap='RdYlGn', subset=['Precision','Recall','F1-Score']),
                use_container_width=True, hide_index=True
            )

            spec = tn / (tn+fp) if (tn+fp) > 0 else 0
            npv  = tn / (tn+fn) if (tn+fn) > 0 else 0
            ppv  = tp / (tp+fp) if (tp+fp) > 0 else 0
            st.markdown("**Clinical Utility Metrics**")
            util_df = pd.DataFrame([
                {'Metric': 'Sensitivity (Recall)', 'Value': f"{r['report']['1']['recall']:.4f}",
                 'Clinical Meaning': 'Readmissions correctly identified'},
                {'Metric': 'Specificity', 'Value': f"{spec:.4f}",
                 'Clinical Meaning': 'Non-readmissions correctly cleared'},
                {'Metric': 'PPV (Precision)', 'Value': f"{ppv:.4f}",
                 'Clinical Meaning': 'Positive predictive value'},
                {'Metric': 'NPV', 'Value': f"{npv:.4f}",
                 'Clinical Meaning': 'Negative predictive value'},
                {'Metric': 'Brier Score', 'Value': f"{r['brier']:.4f}",
                 'Clinical Meaning': 'Probability calibration (lower=better)'},
            ])
            st.dataframe(util_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">ROC & Precision-Recall Curves</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="story-banner">
        <strong>ROC Curve</strong> shows the sensitivity/specificity tradeoff across all thresholds.
        Area Under Curve (AUC) > 0.8 indicates good discrimination.
        <strong>Precision-Recall</strong> is more informative for imbalanced datasets like readmission
        prediction — it directly shows the tradeoff between catching more readmissions (recall)
        and reducing false alarms (precision).
        </div>
        """, unsafe_allow_html=True)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('ROC Curves — All Models', 'Precision-Recall Curves'))
        colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']

        for i, (name, r) in enumerate(results.items()):
            c = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=r['fpr'], y=r['tpr'], mode='lines',
                                      name=f"{name} (AUC={r['roc_auc']:.3f})",
                                      line=dict(color=c, width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=r['recall'], y=r['precision'], mode='lines',
                                      name=f"{name} (AP={r['avg_precision']:.3f})",
                                      line=dict(color=c, width=2), showlegend=False), row=1, col=2)

        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                  line=dict(dash='dash', color='gray', width=1),
                                  showlegend=False), row=1, col=1)

        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate",  row=1, col=1)
        fig.update_xaxes(title_text="Recall",    row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        fig.update_layout(height=500, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">Probability Calibration & Decision Threshold Analysis</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        <strong>Calibration</strong> is critical in clinical AI — a model that says "30% risk"
        should actually see 30% of those patients readmitted. Poorly calibrated models mislead
        care teams. The <strong>decision threshold</strong> (default 0.5) can be tuned to
        prioritize sensitivity (catch more cases) or specificity (fewer false alarms) based on
        clinical workflow capacity.
        </div>
        """, unsafe_allow_html=True)

        sel2 = st.selectbox("Select Model", list(trained_models.keys()), key="cal_model")
        r2 = results[sel2]

        col_cal, col_thresh = st.columns(2)

        with col_cal:
            # Calibration curve
            try:
                frac_pos, mean_pred = calibration_curve(y_test, r2['probabilities'], n_bins=10)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode='lines+markers',
                                          name=sel2, line=dict(color='#1a6eb5', width=3),
                                          marker=dict(size=8)))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                          name='Perfect Calibration',
                                          line=dict(dash='dash', color='gray')))
                fig.update_layout(
                    title="Calibration Curve (Reliability Diagram)",
                    xaxis_title="Mean Predicted Probability",
                    yaxis_title="Fraction of Positives (Actual Rate)",
                    height=400, plot_bgcolor="white"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Calibration curve unavailable: {e}")

        with col_thresh:
            # Threshold analysis
            thresholds = np.arange(0.05, 0.95, 0.05)
            thresh_rows = []
            for t in thresholds:
                preds = (r2['probabilities'] >= t).astype(int)
                tp = ((preds == 1) & (y_test == 1)).sum()
                fp = ((preds == 1) & (y_test == 0)).sum()
                fn = ((preds == 0) & (y_test == 1)).sum()
                tn = ((preds == 0) & (y_test == 0)).sum()
                sens = tp / (tp + fn) if (tp+fn) > 0 else 0
                spec = tn / (tn + fp) if (tn+fp) > 0 else 0
                ppv  = tp / (tp + fp) if (tp+fp) > 0 else 0
                thresh_rows.append({'Threshold': t, 'Sensitivity': sens,
                                    'Specificity': spec, 'PPV': ppv,
                                    'Flagged (%)': (preds.sum()/len(preds))*100})

            thresh_df = pd.DataFrame(thresh_rows)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df['Sensitivity'],
                                      name='Sensitivity', line=dict(color='#e74c3c', width=2)))
            fig.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df['Specificity'],
                                      name='Specificity', line=dict(color='#27ae60', width=2)))
            fig.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df['PPV'],
                                      name='PPV (Precision)', line=dict(color='#1a6eb5', width=2)))
            fig.add_vline(x=0.5, line_dash="dash", annotation_text="Default threshold",
                          line_color="gray")
            fig.update_layout(
                title="Sensitivity / Specificity / PPV vs. Threshold",
                xaxis_title="Decision Threshold", yaxis_title="Metric Value",
                height=400, plot_bgcolor="white", yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)

            thresh_sel = st.slider("Explore threshold", 0.05, 0.95, 0.5, step=0.05)
            row = thresh_df[thresh_df['Threshold'].round(2) == round(thresh_sel, 2)]
            if not row.empty:
                r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                r2c1.metric("Sensitivity", f"{row['Sensitivity'].values[0]:.3f}")
                r2c2.metric("Specificity", f"{row['Specificity'].values[0]:.3f}")
                r2c3.metric("PPV",         f"{row['PPV'].values[0]:.3f}")
                r2c4.metric("Patients Flagged", f"{row['Flagged (%)'].values[0]:.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-title">Feature Importance — What Drives the Predictions?</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="story-banner">
        Feature importance tells clinicians <em>which variables</em> the model uses most.
        For regulatory compliance (FDA SaMD, CMS CDS), model decisions must be explainable.
        Unexpected top features may signal <strong>data leakage</strong> or <strong>bias</strong>
        requiring investigation before clinical deployment.
        </div>
        """, unsafe_allow_html=True)

        fi_model = st.selectbox("Select Model", ['Random Forest','Gradient Boosting','XGBoost','LightGBM'],
                                key='fi_sel')
        mdl = trained_models[fi_model]

        if hasattr(mdl, 'feature_importances_'):
            fi_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': mdl.feature_importances_
            }).sort_values('Importance', ascending=False).head(25)

            c1, c2 = st.columns([3, 2])
            with c1:
                fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                             color='Importance', color_continuous_scale='viridis',
                             title=f"Top 25 Feature Importances — {fi_model}")
                fig.update_layout(height=620, plot_bgcolor="white",
                                  yaxis=dict(autorange="reversed"),
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("**Clinical Interpretation of Top Features**")
                clinical_desc = {
                    'last_bnp': '🫀 BNP — cardiac stress biomarker, strongest HF readmission predictor',
                    'comorbidity_score': '📋 Comorbidity burden — Charlson-based multi-disease score',
                    'last_creatinine': '🧪 Creatinine — renal function; cardiorenal syndrome marker',
                    'length_of_stay': '🏥 LOS — longer stays indicate higher acuity',
                    'prior_admissions_6m': '🔄 Prior admissions — strongest behavioral predictor',
                    'last_sodium': '⚗️ Sodium — hyponatremia linked to poor HF outcomes',
                    'last_hemoglobin': '🩸 Hemoglobin — anemia worsens cardiac output',
                    'hf_med_count': '💊 HF Meds — guideline-directed therapy adherence',
                    'age': '👤 Age — Medicare population, comorbidity accumulation',
                    'has_ckd': '🫘 CKD — chronic kidney disease, cardiorenal risk',
                    'total_meds': '💊 Total meds — polypharmacy risk, complexity',
                    'has_copd': '🫁 COPD — exacerbation/HF overlap',
                }
                for _, row in fi_df.head(10).iterrows():
                    feat = row['Feature'].lower()
                    desc = next((v for k, v in clinical_desc.items() if k in feat), f"📊 {row['Feature']}")
                    st.markdown(f"""
                    <div style="background:#f8f9fa; border-left:3px solid #1a6eb5; 
                                padding:0.4rem 0.7rem; margin:0.3rem 0; 
                                border-radius:0 4px 4px 0; font-size:0.82rem;">
                    <strong>{row['Importance']:.4f}</strong> — {desc}
                    </div>
                    """, unsafe_allow_html=True)

            # Cumulative importance
            fi_df['Cumulative Importance'] = fi_df['Importance'].cumsum()
            fig2 = px.area(fi_df, x='Feature', y='Cumulative Importance',
                           title="Cumulative Feature Importance",
                           labels={'Feature': '', 'Cumulative Importance': 'Cumulative Importance'})
            fig2.add_hline(y=0.8, line_dash="dash", annotation_text="80% coverage", line_color="red")
            fig2.update_xaxes(tickangle=-45)
            fig2.update_layout(height=320, plot_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Save best model ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">💾 Save Best Model for Deployment</div>', unsafe_allow_html=True)

    best_name = comp_df.iloc[0]['Model'] if 'comp_df' in dir() else list(trained_models.keys())[0]
    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"Save {best_name} to disk", type="primary"):
            Path('models').mkdir(exist_ok=True)
            with open('models/best_model.pkl', 'wb') as f:
                pickle.dump(trained_models[best_name], f)
            with open('models/feature_names.pkl', 'wb') as f:
                pickle.dump(X_train.columns.tolist(), f)
            st.success(f"✅ {best_name} saved to models/best_model.pkl")
    with c2:
        st.info(f"📌 Best model: **{best_name}**")
