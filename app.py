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
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.92rem; padding: 0.18rem 0; }
    section[data-testid="stSidebar"] hr { border-color: rgba(220, 232, 245, 0.28); }

    /* ── Sidebar selectbox: BLACK text on WHITE background ── */
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="select"] input,
    section[data-testid="stSidebar"] [data-baseweb="popover"] div,
    section[data-testid="stSidebar"] [data-baseweb="popover"] span {
        color: #111111 !important;
        background-color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdown"] {
        color: #111111 !important;
    }

    /* ── Main area selectbox / inputs ── */
    [data-baseweb="select"] div,
    [data-baseweb="select"] span,
    [data-baseweb="select"] input,
    [data-baseweb="popover"] li,
    [data-baseweb="popover"] div,
    [data-baseweb="popover"] span,
    .stSelectbox div, .stSelectbox span,
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        color: #111111 !important;
        background-color: #ffffff !important;
    }
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        border: 1px solid #aebdcd !important;
        border-radius: 6px;
    }
    [data-baseweb="select"] > div {
        border-color: #aebdcd !important;
        min-height: 2.6rem;
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
    .alert-critical { background:#fff0f0; border:1px solid #e74c3c; border-radius:8px; padding:0.8rem 1rem; margin:0.4rem 0; color:#111 !important; }
    .alert-warning  { background:#fffbea; border:1px solid #f39c12; border-radius:8px; padding:0.8rem 1rem; margin:0.4rem 0; color:#111 !important; }
    .alert-info     { background:#f0f8ff; border:1px solid #1a6eb5; border-radius:8px; padding:0.8rem 1rem; margin:0.4rem 0; color:#111 !important; }
    .alert-success  { background:#f0fff4; border:1px solid #27ae60; border-radius:8px; padding:0.8rem 1rem; margin:0.4rem 0; color:#111 !important; }

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
