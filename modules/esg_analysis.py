import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from common.io import load
from common.ui import demo_banner, init_state, company_selector


def render() -> None:
    init_state()
    demo_banner(["esg_facts", "esg_scores", "company_master", "model_dataset"])
    st.title("ESG Analysis")

    company_df = load("company_master")
    facts_df = load("esg_facts")
    scores_df = load("esg_scores")
    dataset_df = load("model_dataset")

    c1, c2 = st.columns([1, 2])
    with c1:
        cid = company_selector(company_df)
    with c2:
        fy_list = sorted(facts_df["fiscal_year"].unique(), reverse=True)
        selected_fy = st.selectbox("Fiscal Year", fy_list, index=0)

    comp_scores = scores_df[(scores_df["company_id"] == cid) & (scores_df["fiscal_year"] == selected_fy)]
    if len(comp_scores) == 0:
        st.warning("No ESG score for selected company and fiscal year.")
        return

    score_row = comp_scores.iloc[0]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("ESG Total", f"{score_row['esg_total']:.1f}")
    k2.metric("Environment", f"{score_row['esg_environment']:.1f}")
    k3.metric("Social", f"{score_row['esg_social']:.1f}")
    k4.metric("Governance", f"{score_row['esg_governance']:.1f}")
    k5.metric("BRSR Coverage", f"{score_row['esg_coverage']:.0%}")

    tab1, tab2, tab3, tab4 = st.tabs(["Pillar Radar Scorecard", "BRSR Indicators Table", "Peer Comparison", "Historical Trend & Predictive Link"])

    with tab1:
        st.subheader(f"E, S, G Pillar Performance vs Universe Median (FY {selected_fy})")
        median_scores = scores_df[scores_df["fiscal_year"] == selected_fy][["esg_environment", "esg_social", "esg_governance"]].median()

        categories = ["Environment", "Social", "Governance"]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[score_row["esg_environment"], score_row["esg_social"], score_row["esg_governance"]],
            theta=categories,
            fill="toself",
            name=f"{cid} Scores",
            line_color="#2ca02c"
        ))
        fig.add_trace(go.Scatterpolar(
            r=[median_scores["esg_environment"], median_scores["esg_social"], median_scores["esg_governance"]],
            theta=categories,
            fill="toself",
            name="Universe Median",
            line_color="gray"
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Raw BRSR Reported Indicator Facts")
        cfacts = facts_df[(facts_df["company_id"] == cid) & (facts_df["fiscal_year"] == selected_fy)]
        disp_table = cfacts[["indicator_id", "value", "unit_raw", "status", "source_doc", "page", "filed_timestamp", "available_date"]]
        st.dataframe(disp_table, use_container_width=True)

    with tab3:
        st.subheader(f"Peer Total ESG Ranking (FY {selected_fy})")
        peers_fy = scores_df[scores_df["fiscal_year"] == selected_fy].sort_values("esg_total", ascending=False)
        fig_bar = px.bar(
            peers_fy,
            x="company_id",
            y="esg_total",
            color="esg_total",
            color_continuous_scale="Viridis",
            title=f"Total ESG Percentile Rank Across All 10 Firms (FY {selected_fy})"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab4:
        st.subheader("Multi-Year ESG Pillar Trends")
        c_all_scores = scores_df[scores_df["company_id"] == cid].sort_values("fiscal_year")
        fig_trend = px.line(
            c_all_scores,
            x="fiscal_year",
            y=["esg_environment", "esg_social", "esg_governance", "esg_total"],
            markers=True,
            title=f"ESG Pillar Evolution for {cid}"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.subheader("Empirical Scatter: ESG Total vs 63-day Forward Excess Return (Synthetic Demo)")
        st.caption("Demonstrates pipeline feature joining; does not represent real market evidence.")
        if "esg_total" in dataset_df.columns:
            scatter_df = dataset_df.dropna(subset=["esg_total", "y_excess"])
            fig_scat = px.scatter(
                scatter_df,
                x="esg_total",
                y="y_excess",
                color="regime",
                trendline="ols",
                title="Synthetic ESG Total vs Realized 63-day Excess Return"
            )
            st.plotly_chart(fig_scat, use_container_width=True)
