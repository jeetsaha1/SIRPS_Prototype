import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from common.io import load
from common.regime import classify
from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["regime_daily", "index_daily", "model_dataset"])
    st.title("Market Regime Classification")

    regime_df = load("regime_daily")
    index_df = load("index_daily")
    dataset_df = load("model_dataset")

    with st.expander("ℹ️ Causal Regime Detection Rules (Blueprint Section 5)"):
        st.markdown("""
- **Inputs**: NIFTY 50 trailing close, 50-day SMA (`MA50`), 200-day SMA (`MA200`), 252-day high drawdown.
- **Bear Rule**: `Drawdown <= -15%` OR (`Close < MA200` AND `MA50 < MA200` AND `Drawdown <= -10%`).
- **Bull Rule**: `Close > MA200` AND `MA50 > MA200` AND `Drawdown > -10%`.
- **Neutral Rule**: Neither rule satisfied.
- **10-Day Causal Confirmation**: A new raw regime switch takes effect only after persisting for **10 consecutive trading days**. Labels are **never back-dated**.
        """)

    # Interactive sensitivity controls
    c1, c2, c3 = st.columns(3)
    bear_thresh = c1.selectbox("Bear Drawdown Threshold", [-0.10, -0.15, -0.20], index=1)
    confirm_d = c2.selectbox("Confirmation Days", [5, 10, 20], index=1)
    recompute = c3.button("Recompute Causal Regime")

    if recompute:
        res = classify(index_df["nifty_close"], bear_dd=bear_thresh, confirm_days=confirm_d)
        active_df = pd.DataFrame({
            "date": index_df["date"],
            "close": res["close"],
            "ma50": res["ma50"],
            "ma200": res["ma200"],
            "drawdown_252": res["drawdown_252"],
            "raw_regime": res["raw_regime"],
            "confirmed_regime": res["confirmed_regime"],
        })
    else:
        active_df = regime_df

    # Summary counts
    reg_counts = active_df["confirmed_regime"].value_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Trading Days", len(active_df))
    k2.metric("Bull Days", f"{reg_counts.get('Bull', 0)} ({reg_counts.get('Bull', 0)/len(active_df):.1%})")
    k3.metric("Neutral Days", f"{reg_counts.get('Neutral', 0)} ({reg_counts.get('Neutral', 0)/len(active_df):.1%})")
    k4.metric("Bear Days", f"{reg_counts.get('Bear', 0)} ({reg_counts.get('Bear', 0)/len(active_df):.1%})")

    # Chart with regime shading
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=active_df["date"], y=active_df["close"], name="NIFTY 50 Close", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=active_df["date"], y=active_df["ma50"], name="MA 50", line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(x=active_df["date"], y=active_df["ma200"], name="MA 200", line=dict(color="green", width=1)))

    # Add regime background shading
    # Group contiguous blocks of confirmed regimes
    active_df["confirmed_regime"] = active_df["confirmed_regime"].ffill().fillna("Neutral")
    regimes = active_df["confirmed_regime"].values
    dates = active_df["date"].values
    colors = {"Bull": "rgba(46, 204, 113, 0.15)", "Bear": "rgba(231, 76, 60, 0.15)", "Neutral": "rgba(189, 195, 199, 0.15)"}

    start_idx = 0
    vrect_count = 0
    for i in range(1, len(regimes)):
        if (regimes[i] != regimes[start_idx] or i == len(regimes) - 1) and str(regimes[start_idx]) in colors:
            r_type = str(regimes[start_idx])
            if vrect_count < 30 and (dates[i] > dates[start_idx]):
                fig.add_vrect(
                    x0=dates[start_idx],
                    x1=dates[i],
                    fillcolor=colors[r_type],
                    layer="below",
                    line_width=0,
                )
                vrect_count += 1
            start_idx = i

    fig.update_layout(title="NIFTY 50 Path with Confirmed Causal Regime Shading", height=450, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Trailing 252-Day High Drawdown")
        fig_dd = px.area(active_df, x="date", y="drawdown_252", title="Drawdown % (Trigger for Bear Market)")
        st.plotly_chart(fig_dd, use_container_width=True)

    with c_right:
        st.subheader("India VIX Level")
        fig_vix = px.line(index_df, x="date", y="india_vix", title="Market Volatility Gauge (India VIX)")
        st.plotly_chart(fig_vix, use_container_width=True)

    st.subheader("Average 63-Day Excess Return by Firm and Regime (Heatmap)")
    if "regime" in dataset_df.columns:
        pivot_ret = dataset_df.groupby(["company_id", "regime"])["y_excess"].mean().unstack()
        fig_heat = px.imshow(
            pivot_ret,
            labels=dict(x="Market Regime", y="Company", color="Mean Excess Return"),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title="Firm Alpha Across Bull, Neutral, and Bear Environments"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
