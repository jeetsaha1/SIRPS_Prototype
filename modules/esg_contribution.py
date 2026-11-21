import streamlit as st

from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["experiment_results", "placebo_distribution"])
    st.header("ESG Contribution (placeholder)")
    st.info("The generator planted an ESG-linked drift of strength 0.04 per standard deviation per 63 days. Recovering it demonstrates the pipeline, not real ESG value.")
