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

# ─── Enhanced CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit's auto-generated multipage nav (we use our own sidebar) */
    [data-testid="stSidebarNav"] { display: none !important; }

    /* Fonts & global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2342 0%, #1a3a5c 100%);
    }
    section[data-testid="stSidebar"] * { color: #e8edf5 !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0a2342 0%, #1a6eb5 100%);
        color: #ffffff !important;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; color: white; }
    .main-header p  { font-size: 0.95rem; opacity: 0.85; margin: 0.3rem 0 0; color: #cce0ff; }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #1a6eb5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 0.5rem;
    }
    .kpi-card.red   { border-left-color: #e74c3c; }
    .kpi-card.green { border-left-color: #27ae60; }
    .kpi-card.orange{ border-left-color: #f39c12; }
    .kpi-card.purple{ border-left-color: #8e44ad; }

    /* Story banner */
    .story-banner {
        background: #eaf4ff;
        border-left: 4px solid #1a6eb5;
        padding: 0.9rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
        font-size: 0.93rem;
        color: #1a3a5c;
        line-height: 1.6;
    }
    .story-banner strong { color: #0a2342; }

    /* Alert cards */
    .alert-critical {
        background: #fff0f0; border: 1px solid #e74c3c;
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0;
    }
    .alert-warning {
        background: #fffbea; border: 1px solid #f39c12;
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0;
    }
    .alert-info {
        background: #f0f8ff; border: 1px solid #1a6eb5;
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0;
    }

    /* Section title */
    .section-title {
        font-size: 1.15rem; font-weight: 600; color: #0a2342;
        border-bottom: 2px solid #1a6eb5; padding-bottom: 0.3rem;
        margin: 1rem 0 0.8rem;
    }

    /* Benchmark bar */
    .benchmark { display:flex; align-items:center; gap:0.5rem; font-size:0.85rem; color:#555; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; font-size: 0.88rem; }

    /* Footer */
    .footer {
        text-align:center; padding: 1.2rem; color: #888;
        font-size: 0.8rem; border-top: 1px solid #e0e0e0; margin-top: 2rem;
    }
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
