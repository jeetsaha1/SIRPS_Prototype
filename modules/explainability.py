import streamlit as st

from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["feature_importance", "local_contributions"])
    st.header("Explainability (placeholder)")
