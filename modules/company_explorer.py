import streamlit as st

from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["company_master", "prices_daily"])
    st.header("Company Explorer (placeholder)")
