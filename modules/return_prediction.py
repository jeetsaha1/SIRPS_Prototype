import streamlit as st

from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["latest_predictions"])
    st.header("Return Prediction (placeholder)")
