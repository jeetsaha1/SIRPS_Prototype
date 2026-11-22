import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from common.io import load
from common.ui import demo_banner, init_state, company_selector


def render() -> None:
    init_state()
    demo_banner(["company_master", "index_daily", "latest_predictions", "regime_daily"])
    st.title("SIRPS: Sustainable Investment Return Prediction System")
    st.caption("ESG-Aware 63-Trading-Day Excess Return Research Prototype | NIFTY 50 TRI Benchmark | SEED = 42")

    company = load("company_master")
    index = load("index_daily")
    latest = load("latest_predictions")
    regime = load("regime_daily")

    # Top KPI Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Universe", f"{len(company)} Firms", "Core Sustainability")
    c2.metric("Horizon", "63 Days", "Entry t+1, Exit t+64")
    c3.metric("Current Regime", regime["confirmed_regime"].iloc[-1], "10-day Causal Filter")
    c4.metric("Active Models", latest["model"].nunique(), "Ridge, RF, XGBoost")
    c5.metric("Data Status", "Synthetic", "SYNTHETIC_DEMO")

    st.markdown("---")

    # Forecast Overview Controls
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        exp_choice = st.radio("Experiment", ["B (With ESG Features)", "A (Without ESG Baseline)"], horizontal=True)
        exp_key = "B" if "B" in exp_choice else "A"
    with col_ctrl2:
        model_choice = st.selectbox("Primary Model", ["XGBoost", "Ridge", "Random Forest"], index=0)

    # Filter forecasts
    sub_latest = latest[(latest["experiment"] == exp_key) & (latest["model"] == model_choice)]
    if len(sub_latest) == 0:
        sub_latest = latest[latest["experiment"] == exp_key]
    if len(sub_latest) == 0:
        sub_latest = latest

    sub_latest = sub_latest.sort_values("y_hat_excess", ascending=False).reset_index(drop=True)

    # Top 3 & Bottom 3 Highlights
    st.subheader("Quarterly Forward Excess Return Forecasts")
    h_col1, h_col2 = st.columns(2)
    top_3 = sub_latest.head(3)
    bottom_3 = sub_latest.tail(3)

    with h_col1:
        st.markdown("**🟢 Top 3 Ranked Firms (Projected Alpha)**")
        for _, r in top_3.iterrows():
            st.success(f"**{r['company_id']}**: {r['y_hat_excess']:+.2%} excess return (80% band: [{r['lower80']:+.1%}, {r['upper80']:+.1%}])")
    with h_col2:
        st.markdown("**🔴 Bottom 3 Ranked Firms**")
        for _, r in bottom_3.iterrows():
            st.warning(f"**{r['company_id']}**: {r['y_hat_excess']:+.2%} excess return (80% band: [{r['lower80']:+.1%}, {r['upper80']:+.1%}])")

    # Bar chart
    fig_bar = px.bar(
        sub_latest,
        x="company_id",
        y="y_hat_excess",
        color="y_hat_excess",
        color_continuous_scale="RdYlGn",
        title=f"Predicted 63-Day Excess Return by Firm ({model_choice}, Experiment {exp_key})",
        labels={"company_id": "Company", "y_hat_excess": "Predicted Excess Return"}
    )
    fig_bar.add_hline(y=0.0, line_dash="dash", line_color="black")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(sub_latest[["company_id", "date", "y_hat_excess", "y_hat_abs", "lower80", "upper80", "regime", "model_version"]], use_container_width=True)

    # NIFTY 50 Path with Regime Shading
    st.subheader("Benchmark Path: NIFTY 50 Close & Causal Regimes")
    merged_idx = pd.merge(index, regime[["date", "confirmed_regime"]], on="date", how="left")
    merged_idx["confirmed_regime"] = merged_idx["confirmed_regime"].ffill().fillna("Neutral")

    fig_idx = go.Figure()
    fig_idx.add_trace(go.Scatter(x=merged_idx["date"], y=merged_idx["nifty_close"], mode="lines", name="NIFTY 50 Close", line=dict(color="#1f77b4", width=2)))
    
    # Shading
    regimes = merged_idx["confirmed_regime"].values
    dates = merged_idx["date"].values
    colors = {"Bull": "rgba(46, 204, 113, 0.15)", "Bear": "rgba(231, 76, 60, 0.15)", "Neutral": "rgba(189, 195, 199, 0.15)"}
    start_idx = 0
    vrect_count = 0
    for i in range(1, len(regimes)):
        if (regimes[i] != regimes[start_idx] or i == len(regimes) - 1) and str(regimes[start_idx]) in colors:
            r_type = str(regimes[start_idx])
            if vrect_count < 30 and (dates[i] > dates[start_idx]):
                fig_idx.add_vrect(x0=dates[start_idx], x1=dates[i], fillcolor=colors[r_type], layer="below", line_width=0)
                vrect_count += 1
            start_idx = i

    fig_idx.update_layout(height=420, template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_idx, use_container_width=True)

    # Status Cards
    st.subheader("Platform Implementation Status")
    s1, s2, s3 = st.columns(3)
    s1.success("✅ **Implemented**\n- End-to-end data pipeline\n- Causal regime classifier\n- Machine learning models\n- Complete 10-page Streamlit UI")
    s2.info("🔬 **Simulated / Synthetic**\n- 10 fictional green equities\n- Seeded econometric profiles\n- Point-in-time BRSR ESG facts\n- Planted alpha signals")
    s3.warning("🔮 **Future Scope**\n- Real NSE exchange feeds\n- Production XBRL scraping\n- Institutional trade execution\n- Live portfolio API")

    st.caption("⚠️ **Disclaimer**: Academic research prototype operating exclusively on synthetic demonstration data. Not investment advice.")
