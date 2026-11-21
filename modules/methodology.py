import streamlit as st

from common.io import load
from common.schemas import FILES
from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["company_master"])
    st.title("Methodology")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Method",
        "Implemented / Simulated / Future",
        "Data dictionary",
        "Synthetic generation",
        "Limitations",
    ])

    with tab1:
        st.markdown("""
### Target
The primary target is the 63-trading-day log excess return versus NIFTY 50 TRI.
Entry occurs at the close of t+1 and exit at t+64.

### ESG comparison
Experiment A' (no ESG) and B (with ESG) use matched company-date observations.
A null result is retained and reported honestly.
""")
    with tab2:
        st.table({
            "Status": ["Implemented", "Simulated", "Future"],
            "Content": [
                "Schema, mock loader, app shell and P1 pages",
                "Synthetic data and precomputed demonstration results",
                "Real filings, prices and holdout research results",
            ],
        })
    with tab3:
        for name, spec in FILES.items():
            with st.expander(name):
                st.write(f"`{spec['folder']}/{spec['file']}`")
                st.code(", ".join(spec["columns"]))
    with tab4:
        st.write("SEED = 42")
        st.write("Planted momentum signal = 0.02")
        st.write("Planted quality signal = 0.02")
        st.write("Planted ESG drift = 0.04 per standard deviation per 63 days")
    with tab5:
        st.warning("This prototype uses synthetic data. It does not establish that ESG predicts real returns.")
        st.markdown("""
- No public investment advice is provided.
- Sample data is for pipeline demonstration only.
- Real source licensing and ingestion remain future work.
""")
