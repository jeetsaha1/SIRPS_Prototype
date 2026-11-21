import plotly.express as px
import streamlit as st

from common.io import load
from common.ui import demo_banner, init_state, company_selector


def render() -> None:
    init_state()
    demo_banner(["company_master", "index_daily", "latest_predictions"])
    st.title("SIRPS Home")

    company = load("company_master")
    index = load("index_daily")
    latest = load("latest_predictions")

    selected = company_selector(company)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Fictional firms", len(company))
    c2.metric("Forecast horizon", "63 days")
    c3.metric("Latest forecasts", len(latest))
    c4.metric("Models", latest["model"].nunique())
    c5.metric("Data status", "Synthetic")

    st.subheader("Latest forecast")
    table = latest.sort_values("y_hat_excess", ascending=False)
    st.dataframe(table, use_container_width=True)

    fig = px.bar(
        table,
        x="company_id",
        y="y_hat_excess",
        color="experiment",
        title="Latest predicted 63-day excess return",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("NIFTY 50 synthetic path")
    st.line_chart(index.set_index("date")[["nifty_close"]])

    st.subheader("Implementation status")
    a, b, c = st.columns(3)
    a.success("Implemented\nMock-first platform and this prototype shell")
    b.info("Simulated\nSynthetic markets, ESG, fundamentals and model results")
    c.warning("Future\nReal source ingestion and public deployment")

    st.caption("Academic research prototype. Not investment advice.")
