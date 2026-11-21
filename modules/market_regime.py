import streamlit as st

from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["regime_daily"])
    st.header("Market Regime (placeholder)")
