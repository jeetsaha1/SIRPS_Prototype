import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np

from common.io import load
from common.ui import demo_banner, init_state, company_selector


def render() -> None:
    init_state()
    demo_banner(["company_master", "prices_daily", "corporate_actions", "fundamentals_pit", "index_daily"])
    st.title("Company Explorer")

    company_df = load("company_master")
    prices_df = load("prices_daily")
    actions_df = load("corporate_actions")
    funds_df = load("fundamentals_pit")
    index_df = load("index_daily")

    col1, col2 = st.columns([1, 2])
    with col1:
        cid = company_selector(company_df)
    with col2:
        comp_info = company_df[company_df["company_id"] == cid].iloc[0]
        st.subheader(f"{comp_info['name']} ({comp_info['NSE_symbol']})")
        st.caption(f"Sector: **{comp_info['sector']}** | Sub-sector: **{comp_info['sub_sector']}** | ISIN: `{comp_info['ISIN']}`")

    comp_prices = prices_df[prices_df["company_id"] == cid].sort_values("date").reset_index(drop=True)
    comp_actions = actions_df[actions_df["company_id"] == cid].sort_values("ex_date").reset_index(drop=True)
    comp_funds = funds_df[funds_df["company_id"] == cid].sort_values("period_end").reset_index(drop=True)

    # Date range selector
    min_date = comp_prices["date"].min().date()
    max_date = comp_prices["date"].max().date()
    start_d, end_d = st.slider("Observation window", min_value=min_date, max_value=max_date, value=(min_date, max_date))

    mask = (comp_prices["date"].dt.date >= start_d) & (comp_prices["date"].dt.date <= end_d)
    window_prices = comp_prices.loc[mask].reset_index(drop=True)
    if len(window_prices) == 0:
        st.warning("No price data in selected window.")
        return

    # Metric Cards
    latest_row = window_prices.iloc[-1]
    first_row = window_prices.iloc[0]

    tot_ret = (latest_row["adj_close"] / first_row["adj_close"]) - 1.0
    daily_rets = window_prices["adj_close"].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 5 else 0.0

    peak = window_prices["adj_close"].cummax()
    dd = (window_prices["adj_close"] - peak) / peak
    max_dd = dd.min()

    # Beta vs NIFTY
    m_idx = (index_df["date"].dt.date >= start_d) & (index_df["date"].dt.date <= end_d)
    w_index = index_df.loc[m_idx].sort_values("date").set_index("date")["nifty_close"].pct_change().dropna()
    w_stock = window_prices.set_index("date")["adj_close"].pct_change().dropna()
    aligned = pd.concat([w_stock.rename("s"), w_index.rename("i")], axis=1).dropna()
    beta_val = (aligned["s"].cov(aligned["i"]) / aligned["i"].var()) if len(aligned) > 20 else 1.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Adjusted Close", f"Rs {latest_row['adj_close']:.2f}")
    m2.metric("Window Return", f"{tot_ret:.1%}")
    m3.metric("Annualized Vol", f"{ann_vol:.1%}")
    m4.metric("Max Drawdown", f"{max_dd:.1%}")
    m5.metric("Beta vs NIFTY", f"{beta_val:.2f}")

    tab1, tab2, tab3 = st.tabs(["Price & Technical Chart", "Quarterly Fundamentals", "Returns & Risk Analytics"])

    with tab1:
        c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4, c_ctrl5 = st.columns(5)
        chart_type = c_ctrl1.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
        show_ma50 = c_ctrl2.checkbox("MA 50", value=True)
        show_ma200 = c_ctrl3.checkbox("MA 200", value=True)
        show_nifty = c_ctrl4.checkbox("Compare with NIFTY (rebased 100)", value=False)
        show_actions = c_ctrl5.checkbox("Show Corporate Actions", value=True)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])

        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=window_prices["date"],
                open=window_prices["open"],
                high=window_prices["high"],
                low=window_prices["low"],
                close=window_prices["close"],
                name="Raw Price (Close)",
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=window_prices["date"],
                y=window_prices["adj_close"],
                mode="lines",
                name="Adjusted Close",
                line=dict(color="#1f77b4", width=2),
            ), row=1, col=1)

        # MAs
        if show_ma50:
            ma50 = window_prices["adj_close"].rolling(50, min_periods=5).mean()
            fig.add_trace(go.Scatter(x=window_prices["date"], y=ma50, mode="lines", name="MA 50", line=dict(color="orange", width=1.5)), row=1, col=1)
        if show_ma200:
            ma200 = window_prices["adj_close"].rolling(200, min_periods=20).mean()
            fig.add_trace(go.Scatter(x=window_prices["date"], y=ma200, mode="lines", name="MA 200", line=dict(color="green", width=1.5)), row=1, col=1)

        if show_nifty:
            sub_idx = index_df[(index_df["date"].dt.date >= start_d) & (index_df["date"].dt.date <= end_d)].sort_values("date")
            if len(sub_idx) > 0:
                rebased_nifty = (sub_idx["nifty_tri"] / sub_idx["nifty_tri"].iloc[0]) * window_prices["adj_close"].iloc[0]
                fig.add_trace(go.Scatter(x=sub_idx["date"], y=rebased_nifty, mode="lines", name="NIFTY TRI (Rebased)", line=dict(color="gray", dash="dash")), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(x=window_prices["date"], y=window_prices["volume"], name="Volume", marker_color="rgba(100, 150, 200, 0.5)"), row=2, col=1)

        # Corporate actions annotations
        if show_actions:
            sub_acts = comp_actions[(comp_actions["ex_date"].dt.date >= start_d) & (comp_actions["ex_date"].dt.date <= end_d)]
            for _, act in sub_acts.iterrows():
                fig.add_annotation(
                    x=act["ex_date"],
                    y=window_prices.loc[window_prices["date"] >= act["ex_date"], "adj_close"].iloc[0] if (window_prices["date"] >= act["ex_date"]).any() else window_prices["adj_close"].iloc[-1],
                    text=f"{act['action'].upper()} ({act['factor']}x)",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="purple",
                    row=1, col=1
                )

        fig.update_layout(height=520, template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Price Records (Last 10 Days)")
        st.dataframe(window_prices.tail(10)[["date", "open", "high", "low", "close", "adj_close", "volume"]], use_container_width=True)

    with tab2:
        st.subheader("Quarterly Financial Statements (Last 8 Quarters)")
        if len(comp_funds) > 0:
            pivoted = comp_funds.pivot(index="period_end", columns="item", values="value").sort_index(ascending=False).head(8)
            st.dataframe(pivoted, use_container_width=True)

            rev_profit = comp_funds[comp_funds["item"].isin(["revenue", "net_profit"])].pivot(index="period_end", columns="item", values="value").reset_index()
            fig_rev = px.bar(rev_profit, x="period_end", y=["revenue", "net_profit"], barmode="group", title="Revenue and Net Profit (Rs Cr)")
            st.plotly_chart(fig_rev, use_container_width=True)
        else:
            st.info("No quarterly fundamentals found for this entity.")

    with tab3:
        st.subheader("Drawdown & Downside Behaviour")
        fig_dd = px.area(window_prices, x="date", y=dd, title="Drawdown from Peak (252-day window)")
        st.plotly_chart(fig_dd, use_container_width=True)

        st.subheader("Corporate Actions Log")
        st.dataframe(comp_actions, use_container_width=True)
