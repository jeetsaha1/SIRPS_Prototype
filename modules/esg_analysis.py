import streamlit as st

from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["esg_facts", "esg_scores"])
    st.header("ESG Analysis (placeholder)")
