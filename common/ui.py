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
    st.caption("🔬 SIRPS Synthetic Research Prototype (SYNTHETIC_DEMO=1)")
    if using_mock:
        st.warning("⚠️ **Prototype Demo Notice (Mock Mode)**: Loading fallback preview data from `data/mock/`. All figures are synthetic demonstrations and do not represent real-market data or investment advice.")
    else:
        st.info("ℹ️ **Synthetic Research Prototype**: Operating on generated point-in-time data marked `SYNTHETIC_DEMO=1`. All signals, models, and predictions validate pipeline architecture and econometric mechanics, not real market evidence.")


def company_selector(companies) -> str:
    values = sorted(companies["company_id"].unique().tolist())
    return st.selectbox(
        "Company",
        values,
        index=max(0, values.index(st.session_state.company_id)) if st.session_state.company_id in values else 0,
        key="company_id",
    )
