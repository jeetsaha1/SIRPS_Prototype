import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from common.io import load
from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["experiment_results", "placebo_distribution", "metrics_summary"])
    st.title("ESG Incremental Contribution Analysis")

    exp_res = load("experiment_results")
    placebo = load("placebo_distribution")
    metrics = load("metrics_summary")

    # Honesty Panel
    st.warning("⚠️ **Academic Honesty Disclosure**: The synthetic generator planted an ESG-linked drift of strength **0.04 per standard deviation per 63 days**. Recovering this effect demonstrates statistical pipeline validity, not real-world market alpha.")

    c1, c2, c3 = st.columns(3)
    model_choice = c1.selectbox("Model", sorted(exp_res["model"].unique()))
    scope_choice = c2.selectbox("Regime Scope", sorted(exp_res["scope"].unique()))
    show_placebo = c3.checkbox("Show Placebo Distribution (H3)", value=True)

    sub_res = exp_res[(exp_res["model"] == model_choice) & (exp_res["scope"] == scope_choice)]
    if len(sub_res) == 0:
        sub_res = exp_res[exp_res["model"] == model_choice]

    res_row = sub_res.iloc[0]

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Delta MSFE (B - A')", f"{res_row['delta_msfe']:+.6f}", "Negative indicates ESG improves accuracy")
    m2.metric("95% Bootstrap CI", f"[{res_row['ci_low']:+.5f}, {res_row['ci_high']:+.5f}]")
    m3.metric("Clark-West p-value (H1)", f"{res_row['clark_west_p']:.4f}", "Nested comparison vs A'")
    badge = "🟢 Inferential" if res_row["inferential_flag"] else "🟡 Descriptive Only (<60 dates)"
    m4.metric("Evidence Tier", badge, f"{res_row['n_dates']} Dates, {res_row['n_episodes']} Episodes")

    st.markdown("---")

    st.subheader("Primary Hypothesis Test: Experiment A' (No ESG) vs Experiment B (With ESG)")
    st.markdown("""
- **Row Set 2 Matching**: Both models evaluated on **identical company-date observations**, folds, and tuning parameters.
- **Null Hypothesis ($H_0$)**: Adding BRSR ESG indicators does not improve out-of-sample MSFE relative to non-ESG model A'.
- **Test Statistic**: Clark-West adjusted for nested forecast error variance.
    """)

    st.dataframe(sub_res[["experiment_a", "experiment_b", "model", "scope", "delta_msfe", "ci_low", "ci_high", "clark_west_p", "h2_p", "n_dates", "inferential_flag"]], use_container_width=True)

    if show_placebo:
        st.subheader("H3: Placebo Permutation Distribution vs Real ESG Improvement")
        st.markdown("Comparing real Experiment B $\\Delta$MSFE against 30 shuffled-ESG placebo runs (randomly assigning ESG across firms).")

        fig_plac = px.histogram(
            placebo[placebo["experiment"] == "P"],
            x="delta_msfe",
            nbins=15,
            title="Placebo Distribution of ΔMSFE vs Real Experiment B",
            color_discrete_sequence=["#bdc3c7"]
        )
        fig_plac.add_vline(x=res_row["delta_msfe"], line_dash="dash", line_color="green", annotation_text="Real Experiment B ΔMSFE")
        st.plotly_chart(fig_plac, use_container_width=True)

    st.subheader("Regime-Wise ESG Incremental Performance Breakdown")
    regime_cuts = exp_res[exp_res["model"] == model_choice]
    st.dataframe(regime_cuts, use_container_width=True)
