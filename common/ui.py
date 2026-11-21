import streamlit as st

from common.io import is_using_mock


def init_state() -> None:
    if "company_id" not in st.session_state:
        st.session_state.company_id = "C01"
    if "model" not in st.session_state:
        st.session_state.model = "Ridge"
    if "experiment" not in st.session_state:
        st.session_state.experiment = "B"


def demo_banner(names=None) -> None:
    names = names or ["company_master"]
    using_mock = any(is_using_mock(name) for name in names)
    st.caption("Synthetic data demonstrator")
    if using_mock:
        st.warning("Using mock data. This is not real market data or investment evidence.")
    else:
        st.info("Using generated data marked SYNTHETIC_DEMO.")


def company_selector(companies) -> str:
    values = sorted(companies["company_id"].unique().tolist())
    return st.selectbox(
        "Company",
        values,
        index=max(0, values.index(st.session_state.company_id)) if st.session_state.company_id in values else 0,
        key="company_id",
    )
