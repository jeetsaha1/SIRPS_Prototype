import importlib

import pytest

from common.schemas import FILES


PAGES = [
    "home", "company_explorer", "esg_analysis", "market_regime",
    "return_prediction", "model_comparison", "esg_contribution",
    "portfolio_risk", "explainability", "methodology",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_exports_render(page):
    module = importlib.import_module(f"modules.{page}")
    assert callable(getattr(module, "render", None))


def test_schema_count():
    assert len(FILES) == 20


@pytest.mark.parametrize("name", FILES)
def test_schema_has_required_last_column(name):
    assert FILES[name]["columns"][-1] == "data_source"
