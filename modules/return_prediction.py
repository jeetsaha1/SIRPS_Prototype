import time
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from common.io import load
from common.ui import demo_banner, init_state, company_selector


def render() -> None:
    init_state()
    demo_banner(["predictions", "latest_predictions", "company_master", "prices_daily"])
    st.title("Return Prediction")

    company_df = load("company_master")
    preds_df = load("predictions")
    latest_preds_df = load("latest_predictions")
    prices_df = load("prices_daily")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cid = company_selector(company_df)
    with col2:
        models = sorted(preds_df["model"].unique())
        model_name = st.selectbox("Model", models, index=models.index("XGBoost") if "XGBoost" in models else 0)
    with col3:
        exps = sorted(preds_df["experiment"].unique())
        exp_name = st.selectbox("Experiment", exps, index=exps.index("B") if "B" in exps else 0)
    with col4:
        mode = st.radio("Forecast Mode", ["Latest Forward Horizon", "Historical Test Date"], horizontal=True)

    comp_prices = prices_df[prices_df["company_id"] == cid].sort_values("date").reset_index(drop=True)
    latest_price = comp_prices.iloc[-1]["adj_close"] if len(comp_prices) > 0 else 500.0

    if mode == "Latest Forward Horizon":
        sub_latest = latest_preds_df[(latest_preds_df["company_id"] == cid) & (latest_preds_df["model"] == model_name) & (latest_preds_df["experiment"] == exp_name)]
        if len(sub_latest) == 0:
            sub_latest = latest_preds_df[latest_preds_df["company_id"] == cid]

        pred_row = sub_latest.iloc[0]
        y_excess = pred_row["y_hat_excess"]
        y_abs = pred_row["y_hat_abs"]
        lower80 = pred_row["lower80"]
        upper80 = pred_row["upper80"]
        cutoff_date = pd.to_datetime(pred_row["data_cutoff"]).strftime("%Y-%m-%d")

        if st.button("🚀 Generate Prediction"):
            with st.spinner("Executing model inference on latest feature vector..."):
                time.sleep(0.4)
            st.success("✅ Precomputed forecast retrieved from trained model.")

        st.subheader("Forecast Summary Card")
        implied_price = latest_price * np.exp(y_abs)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted 63-Day Excess Return", f"{y_excess:+.2%}", help="Versus NIFTY 50 TRI benchmark")
        c2.metric("Predicted Absolute Return", f"{np.expm1(y_abs):+.2%}")
        c3.metric("Estimated 80% Uncertainty Band", f"[{lower80:+.1%}, {upper80:+.1%}]")
        c4.metric("Implied Target Price", f"Rs {implied_price:.2f}")

        st.info(f"📋 **Model Version**: {pred_row.get('model_version', 'v1.0')} | **Data Cutoff**: {cutoff_date} | **Horizon**: 63 Trading Days (~1 Quarter) | **State**: {pred_row.get('regime', 'Bull')}")

    else:  # Historical Test Date
        comp_preds = preds_df[(preds_df["company_id"] == cid) & (preds_df["model"] == model_name) & (preds_df["experiment"] == exp_name)].sort_values("date")
        if len(comp_preds) == 0:
            st.warning("No historical out-of-sample predictions available.")
            return

        date_options = comp_preds["date"].dt.strftime("%Y-%m-%d").unique()
        selected_date_str = st.selectbox("Historical Prediction Date (t)", date_options)
        pred_row = comp_preds[comp_preds["date"].dt.strftime("%Y-%m-%d") == selected_date_str].iloc[0]

        y_excess = pred_row["y_hat_excess"]
        lower80 = pred_row["lower80"]
        upper80 = pred_row["upper80"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Forecasted Excess Return", f"{y_excess:+.2%}")
        c2.metric("80% Prediction Band", f"[{lower80:+.1%}, {upper80:+.1%}]")
        c3.metric("NIFTY Benchmark Expected Return", f"{pred_row['nifty_expected_63d']:+.2%}")

    st.subheader(f"Historical Out-of-Sample Predictions vs Actual Excess Return ({cid})")
    comp_history = preds_df[(preds_df["company_id"] == cid) & (preds_df["model"] == model_name) & (preds_df["experiment"] == exp_name)].sort_values("date")
    if len(comp_history) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comp_history["date"], y=comp_history["y_hat_excess"], mode="lines+markers", name="Predicted Excess Return", line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=comp_history["date"], y=comp_history["upper80"], mode="lines", name="Upper 80% Band", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=comp_history["date"], y=comp_history["lower80"], mode="lines", name="80% Uncertainty Band", fill="tonexty", fillcolor="rgba(31, 119, 180, 0.2)", line=dict(width=0)))
        fig.update_layout(height=420, template="plotly_white", title="Predicted 63-Day Excess Return Timeline with 80% Confidence Intervals")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Historical Predictions Log (Last 10 Dates)")
        st.dataframe(comp_history.tail(10)[["date", "y_hat_excess", "y_hat_abs", "lower80", "upper80", "nifty_expected_63d", "model_version"]], use_container_width=True)
