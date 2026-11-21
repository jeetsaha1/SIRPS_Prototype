import plotly.express as px
import streamlit as st

from common.io import load
from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["metrics_summary", "predictions", "fold_definitions"])
    st.title("Model Comparison")

    metrics = load("metrics_summary")
    predictions = load("predictions")
    folds = load("fold_definitions")

    experiment = st.selectbox("Experiment", sorted(metrics["experiment"].unique()), key="experiment")
    scope = st.selectbox("Scope", ["All", "Bull", "Neutral", "Bear", "Fold 1", "Fold 2", "Fold 3"])
    metric = st.selectbox("Metric", ["msfe", "rmse", "mae", "r2_zero", "rank_ic_mean", "directional_accuracy", "top3_spread"])

    filtered = metrics[(metrics["experiment"] == experiment) & (metrics["scope"] == scope)]
    st.subheader("Leaderboard")
    st.dataframe(filtered.sort_values(metric), use_container_width=True)

    if len(filtered):
        fig = px.bar(filtered, x="model", y=metric, title=f"{metric.upper()} by model")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predicted vs actual")
    exp_predictions = predictions[predictions["experiment"] == experiment]
    if "y_hat_excess" in exp_predictions:
        fig = px.scatter(exp_predictions, x="y_hat_excess", y="y_hat_excess", title="Precomputed prediction preview")
        st.plotly_chart(fig, use_container_width=True)

    zero_rows = filtered[filtered["model"] == "Zero"]
    if len(zero_rows):
        zero_msfe = float(zero_rows["msfe"].iloc[0])
        best = filtered[filtered["model"] != "Zero"].sort_values("msfe").head(1)
        if len(best):
            best_msfe = float(best["msfe"].iloc[0])
            if best_msfe >= zero_msfe:
                st.warning("No model in this view beats the zero baseline.")
            else:
                st.success(f"The best model has lower MSFE than the zero baseline: {best_msfe:.6f} vs {zero_msfe:.6f}.")

    st.subheader("Fold definitions")
    st.dataframe(folds, use_container_width=True)
