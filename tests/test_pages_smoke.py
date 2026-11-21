import pytest
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
    app = AppTest.from_file("app.py")
    app.run(timeout=10)
    assert not app.exception
