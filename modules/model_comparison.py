import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from common.io import load
from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["metrics_summary", "predictions", "fold_definitions", "model_dataset"])
    st.title("Model Comparison & Benchmark Leaderboard")

    metrics = load("metrics_summary")
    predictions = load("predictions")
    folds = load("fold_definitions")
    dataset = load("model_dataset")

    c1, c2, c3 = st.columns(3)
    experiment = c1.selectbox("Experiment", sorted(metrics["experiment"].unique()), index=2 if "B" in metrics["experiment"].unique() else 0)
    scope = c2.selectbox("Scope", ["All", "Bull", "Neutral", "Bear", "Fold 1", "Fold 2", "Fold 3"], index=0)
    metric = c3.selectbox("Evaluation Metric", ["msfe", "rmse", "mae", "r2_zero", "r2_expanding_mean", "rank_ic_mean", "directional_accuracy", "top3_spread"])

    filtered = metrics[(metrics["experiment"] == experiment) & (metrics["scope"] == scope)].copy()
    if len(filtered) == 0:
        st.warning("No metrics for this experiment and scope.")
        return

    # Leaderboard
    st.subheader(f"Leaderboard: Experiment {experiment} ({scope} Scope)")
    sort_asc = metric in ["msfe", "rmse", "mae"]
    leaderboard = filtered.sort_values(metric, ascending=sort_asc).reset_index(drop=True)
    st.dataframe(leaderboard, use_container_width=True)

    # Verdict Callout: compare best ML model vs Zero baseline
    zero_rows = filtered[filtered["model"] == "Zero"]
    non_zeros = filtered[~filtered["model"].isin(["Zero", "Expanding Mean"])]
    if len(zero_rows) > 0 and len(non_zeros) > 0:
        zero_msfe = float(zero_rows["msfe"].iloc[0])
        best_row = non_zeros.sort_values("msfe").iloc[0]
        best_name = best_row["model"]
        best_msfe = float(best_row["msfe"])

        if best_msfe < zero_msfe:
            pct_beat = (zero_msfe - best_msfe) / zero_msfe
            st.success(f"🏆 **Verdict**: Best model **{best_name}** beats the Zero baseline by **{pct_beat:.2%}** lower MSFE ({best_msfe:.6f} vs {zero_msfe:.6f}).")
        else:
            st.warning(f"⚠️ **Verdict**: No model beats the Zero baseline in this scope (Best {best_name}: {best_msfe:.6f} >= Zero: {zero_msfe:.6f}). Null result retained honestly.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Model Performance by Metric")
        fig_bar = px.bar(
            leaderboard,
            x="model",
            y=metric,
            color="model",
            title=f"{metric.upper()} by Model ({scope})"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.subheader("Rank Information Coefficient (Rank-IC)")
        fig_ic = px.bar(
            leaderboard,
            x="model",
            y="rank_ic_mean",
            color="rank_ic_mean",
            color_continuous_scale="Blues",
            title="Cross-Sectional Rank-IC Mean by Model"
        )
        st.plotly_chart(fig_ic, use_container_width=True)

    # Predicted vs Actual Scatter Plot
    st.subheader("Predicted vs Realized Out-of-Sample Excess Return")
    exp_preds = predictions[predictions["experiment"] == experiment].copy()
    truth_map = dataset.set_index(["company_id", "date"])["y_excess"].to_dict()
    exp_preds["y_true"] = [truth_map.get((c, d), np.nan) for c, d in zip(exp_preds["company_id"], exp_preds["date"])]
    exp_preds = exp_preds.dropna(subset=["y_true"])

    if len(exp_preds) > 0:
        fig_scat = px.scatter(
            exp_preds.sample(min(200, len(exp_preds)), random_state=42),
            x="y_hat_excess",
            y="y_true",
            color="model",
            opacity=0.7,
            title="Out-of-Sample Calibration (Predicted vs True)"
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    st.subheader("Purged & Embargoed Walk-Forward Fold Definitions")
    st.dataframe(folds[folds["experiment"] == experiment], use_container_width=True)
