import pytest
import os
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("page", [
    "home",
    "company_explorer",
    "esg_analysis",
    "market_regime",
    "return_prediction",
    "model_comparison",
    "esg_contribution",
    "portfolio_risk",
    "explainability",
    "methodology",
])
def test_dev_page_smoke(page):
    os.environ["SIRPS_USE_MOCK"] = "1"
    app = AppTest.from_file("app.py")
    app.run(timeout=10)
    labels = {
        "home": "Home",
        "company_explorer": "Company Explorer",
        "esg_analysis": "ESG Analysis",
        "market_regime": "Market Regime",
        "return_prediction": "Return Prediction",
        "model_comparison": "Model Comparison",
        "esg_contribution": "ESG Contribution",
        "portfolio_risk": "Portfolio and Risk",
        "explainability": "Explainability",
        "methodology": "Methodology",
    }
    if app.radio:
        app.radio[0].set_value(labels[page])
        app.run(timeout=10)
    assert not app.exception
