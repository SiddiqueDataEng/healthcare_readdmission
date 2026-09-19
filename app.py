"""
Healthcare Readmission Analytics Dashboard
US Healthcare-Grade Platform — Enhanced Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Page configuration
st.set_page_config(
    page_title="CareIQ — US Healthcare Readmission Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global readability ── */
    :root {
        --ink: #172033;
        --ink-muted: #4b5870;
        --navy: #0a2342;
        --blue: #1a6eb5;
        --green: #1e9b51;
        --green-soft: #e7f9ee;
        --green-deep: #114b2d;
        --surface: #ffffff;
        --surface-soft: #f5f8fc;
        --line: #d7e0ea;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: #f7f9fc !important;
    }
    html, body, [class*="css"], p, span, div, label, li, td, th {
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .main .block-container { padding-top: 1.75rem; padding-bottom: 2rem; }
    .stMarkdown, .stCaption, .stText, [data-testid="stText"] { color: var(--ink); }
    p, li, label, [data-testid="stWidgetLabel"] { color: var(--ink) !important; line-height: 1.55; }
    h1, h2, h3, h4, h5, h6 { color: var(--navy) !important; letter-spacing: 0; }
    small, .stCaption { color: var(--ink-muted) !important; }

    /* ── Sidebar background ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2342 0%, #1a3a5c 100%) !important;
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    /* Sidebar text — keep white on dark bg */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #dce8f5 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label {
        color: #f2f7fc !important;
        background: transparent !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.92rem; padding: 0.18rem 0; }
    section[data-testid="stSidebar"] hr { border-color: rgba(220, 232, 245, 0.28); }

    /* ── Sidebar selectbox: dark green text on pale green background ── */
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="select"] input,
    section[data-testid="stSidebar"] [data-baseweb="popover"] div,
    section[data-testid="stSidebar"] [data-baseweb="popover"] span {
        color: #14532d !important;
        background-color: #d5f5e3 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="select-value"],
    section[data-testid="stSidebar"] [data-baseweb="select"] input,
    section[data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {
        color: #14532d !important;
        -webkit-text-fill-color: #14532d !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdown"] {
        color: #14532d !important;
    }

    /* ── Main area selectbox / inputs ── */
    [data-baseweb="select"] div,
    [data-baseweb="select"] span,
    [data-baseweb="select"] input,
    [data-baseweb="popover"] li,
    [data-baseweb="popover"] div,
    [data-baseweb="popover"] span,
    .stSelectbox div, .stSelectbox span,
    .stSelectbox input {
        color: #111111 !important;
        background-color: #d5f5e3 !important;
    }
    /* Page inputs only: green field with black readable text */
    div[data-testid="stAppViewContainer"] input,
    div[data-testid="stAppViewContainer"] textarea,
    div[data-testid="stAppViewContainer"] [data-baseweb="base-input"],
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background: #dff7e7 !important;
        color: #111111 !important;
        -webkit-text-fill-color: #111111 !important;
        border: 2px solid #1e9b51 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        font-weight: 600 !important;
    }

    div[data-testid="stAppViewContainer"] input::placeholder,
    div[data-testid="stAppViewContainer"] textarea::placeholder {
        color: rgba(17, 17, 17, 0.7) !important;
        -webkit-text-fill-color: rgba(17, 17, 17, 0.7) !important;
    }

    div[data-testid="stNumberInput"] button,
    div[data-testid="stTextInput"] button,
    div[data-testid="stTextArea"] button {
        background: #1e9b51 !important;
        border: 1px solid #1e9b51 !important;
        color: #ffffff !important;
        border-radius: 0 6px 6px 0 !important;
        min-width: 2.1rem !important;
        min-height: 2.1rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stWidgetLabel"] {
        color: #0a2342 !important;
        font-weight: 600 !important;
        margin-bottom: 0.4rem !important;
    }
    [data-baseweb="select"] > div {
        border-color: #aebdcd !important;
        min-height: 2.6rem;
    }

    /* Keep sidebar filter boxes green after the generic input rules above. */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * {
        background: #27ae60 !important;
        background-color: #27ae60 !important;
        border-color: #27ae60 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] *,
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] label {
        color: #f2f7fc !important;
        background: transparent !important;
        -webkit-text-fill-color: #f2f7fc !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] input {
        color: #14532d !important;
        background-color: #d5f5e3 !important;
        -webkit-text-fill-color: #14532d !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] {
        background-color: #d5f5e3 !important;
        border: 1px solid #27ae60 !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] svg {
        fill: #14532d !important;
        color: #14532d !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * {
        background: #27ae60 !important;
        background-color: #27ae60 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [role="combobox"],
    section[data-testid="stSidebar"] [role="combobox"] *,
    section[data-testid="stSidebar"] [aria-haspopup="listbox"],
    section[data-testid="stSidebar"] [aria-haspopup="listbox"] * {
        background: #27ae60 !important;
        background-color: #27ae60 !important;
        border-color: #27ae60 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* Streamlit's visible select shell. Keep this last so it wins over BaseWeb styles. */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"] {
        background: #27ae60 !important;
        background-color: #27ae60 !important;
        background-image: none !important;
        border: 1px solid #27ae60 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    button[kind="secondary"], button[kind="primary"] {
        font-weight: 600 !important;
        border-radius: 6px !important;
    }

    /* ── Metric values ── */
    [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 0.8rem 1rem; }
    [data-testid="stMetricValue"]  { color: #0a2342 !important; font-weight: 700; font-size: 1.75rem !important; }
    [data-testid="stMetricLabel"]  { color: #38465d !important; font-weight: 600; }
    [data-testid="stMetricDelta"]  { font-size: 0.82rem !important; }

    /* ── Tables & dataframes ── */
    .dataframe td, .dataframe th { color: #172033 !important; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
    .stTabs [data-baseweb="tab"]      { border-radius: 6px 6px 0 0; font-size: 0.9rem; color: #38465d !important; padding: 0.55rem 0.8rem; }
    .stTabs [aria-selected="true"]    { background: #e8f0fe !important; color: #0a2342 !important; font-weight: 700; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #1a6eb5 !important; height: 3px; }

    /* ── Expanders and bordered content ── */
    [data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }
    [data-testid="stExpander"] summary p { color: var(--navy) !important; font-weight: 600; }

    /* ── Main header ── */
    .main-header {
        background: linear-gradient(135deg, #0a2342 0%, #1a6eb5 100%);
        padding: 1.4rem 2rem; border-radius: 12px; margin-bottom: 1.4rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; color: #ffffff !important; margin: 0; }
    .main-header p  { font-size: 0.9rem; color: #cce0ff !important; margin: 0.3rem 0 0; }

    /* ── Story banner ── */
    .story-banner {
        background: #eaf4ff; border-left: 4px solid #1a6eb5;
        padding: 0.85rem 1.2rem; border-radius: 0 8px 8px 0;
        margin-bottom: 1rem; font-size: 0.92rem;
        color: #1a3a5c !important; line-height: 1.65;
    }
    .story-banner strong { color: #0a2342 !important; }

    /* ── Alert boxes ── */
    .alert-critical { background:#ebf9f0; border:2px solid #1e9b51; border-radius:8px; padding:0.8rem 1rem; margin:0.4rem 0; color:#0d3d25 !important; box-shadow: 0 0 0 1px rgba(30,155,81,0.08); }
    .alert-warning  { background:#ebf9f0; border:2px solid #1e9b51; border-radius:8px; padding:0.8rem 1rem; margin:0.4rem 0; color:#0d3d25 !important; box-shadow: 0 0 0 1px rgba(30,155,81,0.08); }
    .alert-info     { background:#ebf9f0; border:2px solid #1e9b51; border-radius:8px; padding:0.8rem 1rem; margin:0.4rem 0; color:#0d3d25 !important; box-shadow: 0 0 0 1px rgba(30,155,81,0.08); }
    .alert-success  { background:#ebf9f0; border:2px solid #1e9b51; border-radius:8px; padding:0.8rem 1rem; margin:0.4rem 0; color:#0d3d25 !important; box-shadow: 0 0 0 1px rgba(30,155,81,0.08); }

    /* ── Section title ── */
    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #0a2342 !important;
        border-bottom: 2px solid #1a6eb5; padding-bottom: 0.3rem;
        margin: 1rem 0 0.8rem;
    }

    /* ── SQL editor override ── */
    .stCodeBlock, .stCode { color: #111 !important; }
    pre code { color: #1a1a2e !important; background: #f4f6f9 !important; }

    /* ── Footer ── */
    .footer {
        text-align:center; padding:1.1rem; color:#666 !important;
        font-size:0.8rem; border-top:1px solid #e0e0e0; margin-top:2rem;
    }

    /* Hide Streamlit default multipage nav */
    [data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 CareIQ — Hospital Readmission Intelligence Platform</h1>
    <p>US Healthcare Analytics · Heart Failure Cohort · 30-Day Readmission Prevention · Powered by AI/ML</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 CareIQ Navigation")
    st.markdown("---")
    
    page = st.radio(
        "Select Module",
        [
            "🏠 Executive Summary",
            "📊 Clinical Overview & EDA",
            "📈 Advanced Analytics",
            "🔬 SQL Intelligence",
            "🤖 AI / ML Models",
            "🧠 Model Explainability",
            "🎯 Patient Risk Predictions",
            "📋 Data Explorer",
            "📦 Data Management",
        ]
    )
    
    st.markdown("---")
    
    # Hospital context selector
    st.markdown("**🏨 Hospital Context**")
    hospital = st.selectbox("Facility", ["All Facilities", "Memorial General", "St. Luke's Medical", "Valley Health System"])
    year_filter = st.selectbox("Program Year", ["2024", "2023", "2022"])
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#aab8cc; line-height:1.7;'>
    📌 <b>CMS Penalty Threshold:</b> &gt;1.0% excess readmission<br>
    📌 <b>National HF Readmit Rate:</b> ~22%<br>
    📌 <b>HRRP Program:</b> Active FY2024<br>
    📌 <b>Data:</b> Synthetic EHR · HIPAA Safe Harbor
    </div>
    """, unsafe_allow_html=True)

# ─── Data loader ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    full_data  = pd.read_csv('data/processed/full_dataset.csv')
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data  = pd.read_csv('data/processed/test_data.csv')
    for df in [full_data, train_data, test_data]:
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype(float)
    return full_data, train_data, test_data

try:
    full_data, train_data, test_data = load_data()
except FileNotFoundError:
    st.error("⚠️ Data files not found. Please run `START_HERE.bat` to generate data.")
    st.stop()


# ─── Plain-language chart guidance ───────────────────────────────────────────
_original_plotly_chart = st.plotly_chart


def _chart_guidance(title):
    """Return plain-language interpretation text for a chart title."""
    title_lower = title.lower()
    if any(word in title_lower for word in ["correlation", "relationship"]):
        return (
            "Issue: variables may move together. Reason: shared clinical or utilization patterns can affect both. "
            "Use: treat this as an association, not proof that one variable causes another. "
            "Impact: it helps prioritize questions for deeper clinical review."
        )
    if any(word in title_lower for word in ["model", "roc", "precision", "recall", "calibration", "shap", "feature importance"]):
        return (
            "Issue: a prediction may miss some patients or flag patients unnecessarily. Reason: models learn patterns from historical data. "
            "Use: check recall, precision, calibration, and the number of positive cases before trusting it. "
            "Impact: the result can support research review, not replace clinical judgment."
        )
    if any(word in title_lower for word in ["cost", "saving", "penalty", "financial"]):
        return (
            "Issue: financial results depend on assumptions. Reason: cost, payer, and volume inputs may not represent a real hospital. "
            "Use: change the inputs to test a hypothetical scenario. "
            "Impact: this shows possible scale, not validated savings or an actual penalty."
        )
    if any(word in title_lower for word in ["trend", "time", "monthly", "year"]):
        return (
            "Issue: rates can change over time. Reason: patient mix and the number of records may differ between periods. "
            "Use: read the rate together with its record count. "
            "Impact: it can identify patterns worth investigating, not prove improvement by itself."
        )
    if any(word in title_lower for word in ["risk", "tier", "readmission"]):
        return (
            "Issue: some records may have a higher observed or predicted outcome rate. Reason: clinical severity and prior utilization differ between groups. "
            "Use: compare both the rate and the group size. "
            "Impact: it can help prioritize analysis, but it is not a diagnosis or automatic care instruction."
        )
    if any(word in title_lower for word in ["distribution", "histogram", "box", "violin", "population"]):
        return (
            "Issue: patient values may be concentrated, spread out, or skewed. Reason: patients have different clinical profiles. "
            "Use: look for the typical range and unusual values before comparing groups. "
            "Impact: it explains who is represented in the cohort and where data quality needs attention."
        )
    return (
        "Issue: the visual summarizes a pattern in this dataset. Reason: records differ in outcomes, characteristics, or data quality. "
        "Use: hover over points or bars and check the sample size. "
        "Impact: it supports understanding and investigation, not a causal or clinical conclusion."
    )


def _add_chart_guidance(figure, **kwargs):
    """Add accessible hover text and a visible interpretation to every Plotly chart."""
    title = ""
    if getattr(figure.layout, "title", None):
        title = figure.layout.title.text or ""
    guidance = _chart_guidance(title)
    for trace in figure.data:
        trace_type = getattr(trace, "type", "")
        if trace_type == "pie":
            trace.hovertemplate = (
                "<b>%{label}</b><br>Value: %{value}<br>Share: %{percent}<br><br>"
                + guidance.replace("\n", "<br>")
                + "<extra></extra>"
            )
        elif trace_type == "heatmap":
            trace.hovertemplate = (
                "X: %{x}<br>Y: %{y}<br>Value: %{z:.3f}<br><br>"
                + guidance.replace("\n", "<br>")
                + "<extra></extra>"
            )
        elif trace_type not in ["indicator", "table"]:
            trace.hovertemplate = (
                "<b>%{fullData.name}</b><br>X: %{x}<br>Y: %{y}<br><br>"
                + guidance.replace("\n", "<br>")
                + "<extra></extra>"
            )
    result = _original_plotly_chart(figure, **kwargs)
    st.caption(f"How to read this visual: {guidance}")
    return result


st.plotly_chart = _add_chart_guidance

_original_pyplot = st.pyplot


def _explained_pyplot(*args, **kwargs):
    """Add plain-language guidance to Matplotlib visuals such as SHAP plots."""
    result = _original_pyplot(*args, **kwargs)
    st.caption(
        "How to read this visual: Issue: a feature may appear influential in the model. "
        "Reason: the model used it to change its estimate for the selected records. "
        "Use: compare direction and size with clinical context. "
        "Impact: this explains model behavior, but does not prove that the feature caused readmission."
    )
    return result


st.pyplot = _explained_pyplot


from streamlit.delta_generator import DeltaGenerator

_original_metric = DeltaGenerator.metric
_original_root_metric = st.metric


def _metric_guidance(label):
    """Return a short non-technical explanation for a KPI label."""
    label_lower = str(label).lower()
    if "rate" in label_lower or "readmit" in label_lower:
        return "What it means: a count or percentage of readmission outcomes. Check the number of records behind it before interpreting the result."
    if "cost" in label_lower or "saving" in label_lower or "penalty" in label_lower:
        return "What it means: a cost or scenario amount based on entered assumptions, not a verified financial result."
    if "risk" in label_lower or "score" in label_lower:
        return "What it means: a summary score or estimate. It supports review and does not diagnose a patient or prescribe care."
    if "sample" in label_lower or "patient" in label_lower or "cohort" in label_lower or "record" in label_lower:
        return "What it means: the number of records included in this calculation. Small groups can make rates unstable."
    return "What it means: an average or count calculated from the selected records. Use it to understand the cohort, not as proof of cause or improvement."


def _explained_metric(self, label, value, delta=None, delta_color="normal", *, help=None, **kwargs):
    if help is None:
        help = _metric_guidance(label)
    return _original_metric(
        self, label, value, delta, delta_color=delta_color, help=help, **kwargs
    )


def _explained_root_metric(label, value, delta=None, delta_color="normal", *, help=None, **kwargs):
    if help is None:
        help = _metric_guidance(label)
    return _original_root_metric(
        label, value, delta, delta_color=delta_color, help=help, **kwargs
    )


DeltaGenerator.metric = _explained_metric
st.metric = _explained_root_metric

# ─── Page routing ─────────────────────────────────────────────────────────────
if page == "🏠 Executive Summary":
    from page_modules import executive_summary
    executive_summary.show(full_data, train_data, test_data)

elif page == "📊 Clinical Overview & EDA":
    from page_modules import overview
    overview.show(full_data, train_data, test_data)

elif page == "📈 Advanced Analytics":
    from page_modules import analytics
    analytics.show(full_data, train_data, test_data)

elif page == "🔬 SQL Intelligence":
    from page_modules import sql_explorer
    sql_explorer.show()

elif page == "🤖 AI / ML Models":
    from page_modules import ml_models
    ml_models.show(train_data, test_data)

elif page == "🧠 Model Explainability":
    from page_modules import explainability
    explainability.show(train_data, test_data)

elif page == "🎯 Patient Risk Predictions":
    from page_modules import predictions
    predictions.show(train_data, test_data)

elif page == "📋 Data Explorer":
    from page_modules import data_explorer
    data_explorer.show(full_data)

elif page == "📦 Data Management":
    from page_modules import data_management
    data_management.show()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CareIQ Healthcare Intelligence Platform &nbsp;|&nbsp;
    Synthetic EHR Data &nbsp;|&nbsp;
    Heart Failure Readmission Prevention &nbsp;|&nbsp;
    CMS HRRP Aligned &nbsp;|&nbsp;
    Built with Streamlit · Python · XGBoost · SHAP
</div>
""", unsafe_allow_html=True)
