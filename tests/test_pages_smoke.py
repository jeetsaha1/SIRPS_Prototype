import os
import pytest
from streamlit.testing.v1 import AppTest

PAGES = [
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
]

LABELS = {
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


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("use_mock", ["1", "0"])
def test_page_smoke(page, use_mock):
    """Verifies every page loads successfully in both mock mode and real synthetic mode."""
    os.environ["SIRPS_USE_MOCK"] = use_mock
    app = AppTest.from_file("app.py")
    timeout = 30 if page == "explainability" else 20
    app.run(timeout=timeout)

    target_label = LABELS[page]
    nav_radio = None
    for r in app.radio:
        if target_label in r.options:
            nav_radio = r
            break
    if nav_radio and nav_radio.value != target_label:
        nav_radio.set_value(target_label)
        app.run(timeout=timeout)

    assert not app.exception
