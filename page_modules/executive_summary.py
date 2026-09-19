"""
Executive summary for the synthetic heart-failure readmission cohort.

All metrics on this page are descriptive unless explicitly labeled as a
reference or hypothetical scenario.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


EXTERNAL_HF_REFERENCE = 22.0


def wilson_interval(successes, observations, confidence_z=1.96):
    """Return an approximate Wilson interval for a binary proportion."""
    if observations == 0:
        return 0.0, 0.0
    proportion = successes / observations
    denominator = 1 + confidence_z ** 2 / observations
    center = (proportion + confidence_z ** 2 / (2 * observations)) / denominator
    margin = confidence_z * np.sqrt(
        proportion * (1 - proportion) / observations
        + confidence_z ** 2 / (4 * observations ** 2)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def show(full_data, train_data, test_data):
    st.markdown("## Executive Summary - Synthetic Cohort")

    total_patients = len(full_data)
    total_readmits = int(full_data["readmitted_30d"].sum())
    readmit_rate = full_data["readmitted_30d"].mean() * 100
    avg_los = full_data["length_of_stay"].mean()
    avg_age = full_data["age"].mean()
    reference_gap = readmit_rate - EXTERNAL_HF_REFERENCE
    ci_low, ci_high = wilson_interval(total_readmits, total_patients)

    st.markdown(f"""
    <div class="story-banner">
    <strong>Dataset context:</strong> This page describes a synthetic exploratory cohort of
    <strong>{total_patients:,} index-admission records</strong> with
    <strong>{total_readmits:,} observed 30-day readmissions</strong>.
    The external heart-failure reference is context only. These results are not hospital
    performance, prevented readmissions, HRRP eligibility, or financial impact.
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        f"Sparse outcome warning: {total_readmits} readmission events were observed. "
        f"The rate is {readmit_rate:.1f}% with an approximate 95% interval of "
        f"{ci_low * 100:.1f}% to {ci_high * 100:.1f}%. Interpret subgroup and model results cautiously."
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Index Admissions", f"{total_patients:,}")
    c2.metric("Observed 30-Day Rate", f"{readmit_rate:.1f}%")
    c3.metric("Observed Readmissions", f"{total_readmits:,}")
    c4.metric("Average LOS", f"{avg_los:.1f} days")
    c5.metric("Average Age", f"{avg_age:.0f} yrs")
    c6.metric(
        "Difference vs Reference",
        f"{reference_gap:+.1f} pp",
        help="Observed cohort rate minus the external 22% reference. This is not prevented readmissions."
    )

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-title">Observed Rate vs External Reference</div>', unsafe_allow_html=True)
        rates = pd.DataFrame({
            "Group": ["This synthetic cohort", "External HF reference"],
            "Rate": [readmit_rate, EXTERNAL_HF_REFERENCE],
        })
        fig = px.bar(
            rates,
            x="Group",
            y="Rate",
            text=rates["Rate"].map(lambda value: f"{value:.1f}%"),
            color="Group",
            color_discrete_sequence=["#1a6eb5", "#f39c12"],
            labels={"Rate": "30-day readmission rate (%)"},
            title="Contextual comparison only",
        )
        fig.update_traces(textposition="outside", showlegend=False)
        fig.update_layout(height=380, yaxis_range=[0, 30], plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The 22% reference is not a risk-adjusted CMS result for this synthetic cohort.")

    with col_right:
        st.markdown('<div class="section-title">Hypothetical Cost Scenarios</div>', unsafe_allow_html=True)
        st.caption("Arithmetic demonstrations only. These are not HRRP calculations, annual forecasts, or validated savings.")
        cost_per_readmit = st.number_input(
            "Illustrative cost per readmission ($)", value=15000, step=1000, min_value=0
        )
        observed_cost = total_readmits * cost_per_readmit
        st.info(
            f"Observed-cohort scenario: {total_readmits} x ${cost_per_readmit:,} = "
            f"${observed_cost:,.0f}."
        )
        reduction_target = st.slider("Hypothetical reduction (%)", 5, 50, 20)
        comparable_events = total_readmits * (1 - reduction_target / 100)
        illustrative_difference = total_readmits * reduction_target / 100 * cost_per_readmit
        st.success(
            f"If a comparable cohort had a {reduction_target}% reduction, it would have "
            f"about {comparable_events:g} events, an illustrative difference of "
            f"${illustrative_difference:,.0f}."
        )

    st.markdown("---")
    st.markdown('<div class="section-title">Descriptive Cohort Metrics</div>', unsafe_allow_html=True)
    st.caption("These summaries describe this dataset; they are not hospital quality scores or HRRP measures.")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Observed Rate", f"{readmit_rate:.1f}%", help=f"Approximate 95% interval: {ci_low * 100:.1f}% to {ci_high * 100:.1f}%")
    metric_col2.metric("Average Length of Stay", f"{avg_los:.1f} days")
    metric_col3.metric("Average Comorbidity Score", f"{full_data['comorbidity_score'].mean():.1f}")

    st.markdown("---")
    st.markdown('<div class="section-title">Exploratory Rule-Based Flags</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="story-banner">
    This transparent score groups records using age, BNP, length of stay, comorbidity,
    selected conditions, and prior admissions. It is not a validated LACE, HOSPITAL,
    or machine-learning score and must not determine patient care.
    </div>
    """, unsafe_allow_html=True)

    flagged = full_data.copy()
    flagged["exploratory_score"] = 0
    flagged["exploratory_score"] += (flagged["age"] > 75).astype(int)
    flagged["exploratory_score"] += (flagged["last_bnp"] > 1000).astype(int) * 2
    flagged["exploratory_score"] += (flagged["comorbidity_score"] >= 3).astype(int) * 2
    flagged["exploratory_score"] += (flagged["length_of_stay"] > 7).astype(int)
    flagged["exploratory_score"] += flagged["has_ckd"]
    flagged["exploratory_score"] += flagged["has_copd"]
    flagged["exploratory_score"] += (flagged["prior_admissions_6m"] > 0).astype(int) * 2
    flagged["exploratory_tier"] = pd.cut(
        flagged["exploratory_score"],
        bins=[-1, 2, 4, 6, 100],
        labels=["Low", "Moderate", "High", "Very High"],
    )

    tier_summary = flagged.groupby("exploratory_tier", observed=True).agg(
        records=("readmitted_30d", "count"),
        observed_readmits=("readmitted_30d", "sum"),
        observed_rate=("readmitted_30d", "mean"),
    ).reset_index()
    tier_summary["observed_rate"] *= 100
    example_reviews = {
        "Low": "No additional example action",
        "Moderate": "Review follow-up needs",
        "High": "Consider care-team review",
        "Very High": "Consider multidisciplinary review",
    }
    tier_summary["example_review"] = tier_summary["exploratory_tier"].astype(str).map(example_reviews)

    left, right = st.columns(2)
    with left:
        fig = px.pie(
            tier_summary,
            values="records",
            names="exploratory_tier",
            hole=0.45,
            title="Records by exploratory tier",
            color_discrete_sequence=["#27ae60", "#f39c12", "#e67e22", "#e74c3c"],
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(
            tier_summary,
            x="exploratory_tier",
            y="observed_rate",
            text=tier_summary["observed_rate"].map(lambda value: f"{value:.1f}%"),
            title="Observed rate by exploratory tier",
            labels={"observed_rate": "Observed 30-day rate (%)", "exploratory_tier": "Tier"},
            color="exploratory_tier",
            color_discrete_sequence=["#27ae60", "#f39c12", "#e67e22", "#e74c3c"],
        )
        fig.update_traces(textposition="outside", showlegend=False)
        fig.update_layout(height=340, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        tier_summary.rename(columns={
            "exploratory_tier": "Exploratory Tier",
            "records": "Records",
            "observed_readmits": "Observed Readmissions",
            "observed_rate": "Observed Rate (%)",
            "example_review": "Example Review",
        }).style.format({"Observed Rate (%)": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("No clinical action is implied by the exploratory flags.")
