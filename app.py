import streamlit as st

from common.ui import init_state


st.set_page_config(page_title="SIRPS Prototype", layout="wide")
init_state()

pages = {
    "Home": "home",
    "Company Explorer": "company_explorer",
    "ESG Analysis": "esg_analysis",
    "Market Regime": "market_regime",
    "Return Prediction": "return_prediction",
    "Model Comparison": "model_comparison",
    "ESG Contribution": "esg_contribution",
    "Portfolio and Risk": "portfolio_risk",
    "Explainability": "explainability",
    "Methodology": "methodology",
}

selection = st.sidebar.radio("SIRPS", list(pages.keys()))
module_name = pages[selection]
import importlib
importlib.import_module(f"modules.{module_name}").render()
