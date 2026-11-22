import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common.io import load
from common.portfolio_engine import simulate_portfolio
from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["portfolio_default", "predictions", "prices_daily", "index_daily", "regime_daily", "company_master"])
    st.title("Portfolio Construction & Staggered-Cohort Backtesting")

    preds = load("predictions")
    prices = load("prices_daily")
    index = load("index_daily")
    regime = load("regime_daily")
    comp = load("company_master")
    default_sim = load("portfolio_default")

    with st.expander("ℹ️ Staggered-Cohort Investment Methodology (Blueprint Section 11 & Roadmap Phase 4)"):
        st.markdown("""
- **Cohort Formation**: Every **21 trading days**, form a new cohort selecting the **top-k** firms by out-of-sample predicted excess return.
- **Holding Period**: Each cohort entered at close of $t+1$ and exited at close of $t+64$ (held exactly **63 trading days**). In steady state, 3 cohorts overlap.
- **Weighting & Costs**: Equal-weight ($1/k$) per cohort; **2 × cost** amortized over the 63 days ($\text{daily cost} = 2 \times \text{cost} / 63$). Zero cost equals gross return.
- **Portfolio Value**: Daily portfolio return is the equal-weight average across all currently active cohorts.
- **Benchmarks**: NIFTY 50 Total Return Index (TRI), Equal-Weight Universe, and 200 Random Top-k draws (5th–95th percentile).
        """)

    # Interactive Controls
    c1, c2, c3, c4 = st.columns(4)
    model_choice = c1.selectbox("Forecast Model", sorted(preds["model"].unique()), index=sorted(preds["model"].unique()).index("XGBoost") if "XGBoost" in preds["model"].unique() else 0)
    k_choice = c2.slider("Cohort Size (Top-k)", min_value=2, max_value=5, value=3)
    cost_choice = c3.slider("One-Way Cost (%)", min_value=0.0, max_value=0.5, value=0.3, step=0.05) / 100.0
    initial_capital = c4.number_input("Initial Investment (Rs)", value=100000, step=10000)

    # Benchmark checkboxes
    b1, b2, b3 = st.columns(3)
    show_nifty = b1.checkbox("Benchmark: NIFTY 50 TRI", value=True)
    show_equal = b2.checkbox("Benchmark: Equal-Weight Universe", value=True)
    show_random = b3.checkbox("Benchmark: Random Top-k Band (5-95%)", value=True)

    run_sim = st.button("⚡ Run Cohort Backtest Simulation")

    # Run simulation or generate default structure
    sim_res = simulate_portfolio(
        preds[preds["model"] == model_choice],
        prices,
        index,
        regime=regime,
        company_master=comp,
        k=k_choice,
        cost=cost_choice,
    )
    daily_df = sim_res["daily"]
    metrics = sim_res["metrics"]
    holdings_df = sim_res["holdings"]
    by_regime_df = sim_res["by_regime"]

    # Summary Metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Strategy Annual Return", f"{metrics['annualized_return']:.1%}")
    m2.metric("Annualized Volatility", f"{metrics['annualized_volatility']:.1%}")
    m3.metric("Sharpe Ratio (Rf=6.5%)", f"{metrics['sharpe_ratio']:.2f}")
    m4.metric("Maximum Drawdown", f"{metrics['max_drawdown']:.1%}")
    m5.metric("Annual Turnover", f"{metrics['turnover']:.1f}x")
    m6.metric("Total Trading Costs", f"Rs {metrics['costs_paid']:,.0f}")

    # Equity Curve in Rupees
    st.subheader(f"Portfolio Growth from Rs {initial_capital:,.0f}")
    scaled_df = daily_df.copy()
    for col in ["strategy", "nifty_tri", "equal_weight", "random_mean"]:
        scaled_df[col] = (scaled_df[col] / scaled_df[col].iloc[0]) * initial_capital

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=scaled_df["date"], y=scaled_df["strategy"], mode="lines", name="SIRPS Strategy", line=dict(color="#2ca02c", width=2.5)))
    if show_nifty:
        fig_eq.add_trace(go.Scatter(x=scaled_df["date"], y=scaled_df["nifty_tri"], mode="lines", name="NIFTY 50 TRI", line=dict(color="#1f77b4", width=1.5, dash="dash")))
    if show_equal:
        fig_eq.add_trace(go.Scatter(x=scaled_df["date"], y=scaled_df["equal_weight"], mode="lines", name="Equal-Weight Universe", line=dict(color="gray", width=1.2)))
    
    if show_random and "random_p05" in scaled_df.columns and "random_p95" in scaled_df.columns:
        p05 = (scaled_df["random_p05"] / scaled_df["random_p05"].iloc[0]) * initial_capital
        p95 = (scaled_df["random_p95"] / scaled_df["random_p95"].iloc[0]) * initial_capital
        fig_eq.add_trace(go.Scatter(x=scaled_df["date"], y=p95, mode="lines", line=dict(width=0), showlegend=False))
        fig_eq.add_trace(go.Scatter(x=scaled_df["date"], y=p05, mode="lines", fill="tonexty", fillcolor="rgba(200, 200, 200, 0.2)", name="Random Top-k Band (5-95%)", line=dict(width=0)))

    fig_eq.update_layout(height=480, template="plotly_white", yaxis_title="Rupees (INR)")
    st.plotly_chart(fig_eq, use_container_width=True)

    # Drawdown Curve
    st.subheader("Strategy Drawdown Profile")
    strat_vals = daily_df["strategy"].values
    running_peak = np.maximum.accumulate(strat_vals)
    dd_curve = (strat_vals - running_peak) / running_peak
    fig_dd = px.area(daily_df, x="date", y=dd_curve, title="Strategy Drawdown Over Time")
    st.plotly_chart(fig_dd, use_container_width=True)

    # Performance by Start Regime & Latest Cohort Holdings
    r_col, h_col = st.columns(2)
    with r_col:
        st.subheader("Performance by Regime")
        if not by_regime_df.empty:
            st.dataframe(by_regime_df.drop(columns=["data_source"], errors="ignore"), use_container_width=True)
        else:
            st.info("Regime breakdown not available.")

    with h_col:
        st.subheader("Latest Active Cohort Holdings")
        if not holdings_df.empty:
            st.dataframe(holdings_df.drop(columns=["data_source"], errors="ignore"), use_container_width=True)
        else:
            st.info("Cohort holdings not available.")

    # Download CSV
    csv_bytes = daily_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Simulation Daily Track Record (CSV)", data=csv_bytes, file_name="sirps_backtest_daily.csv", mime="text/csv")
