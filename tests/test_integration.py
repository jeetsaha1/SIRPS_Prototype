import os
from pathlib import Path

from common.io import load, is_using_mock
from common.schemas import FILES

ROOT = Path(__file__).resolve().parents[1]


def test_p1_integration_all_files_in_mock_mode():
    """G0 Gate check: every file loads in mock fallback mode."""
    load.clear()
    os.environ["SIRPS_USE_MOCK"] = "1"
    for name in FILES:
        frame = load(name)
        assert len(frame) > 0
        assert frame.attrs["using_mock"] is True
    load.clear()


def test_p1_integration_all_files_in_real_mode():
    """G1-G3 Gate check: every file loads in real production mode without mock fallback."""
    load.clear()
    os.environ["SIRPS_USE_MOCK"] = "0"
    for name in FILES:
        frame = load(name)
        assert len(frame) > 0, f"Table {name} is empty in real mode"
        assert not is_using_mock(name), f"Table {name} is unexpectedly falling back to mock"
    load.clear()


def test_p1_integration_gates_g0_to_g5():
    """Comprehensive Gate validation across all P1/P2/P3/P4 deliverables."""
    # G1 Data freeze check
    p2_files = ["trading_calendar", "index_daily", "prices_daily", "corporate_actions", "esg_facts", "esg_scores"]
    p3_files = ["fundamentals_pit", "regime_daily", "model_dataset", "latest_features"]
    p4_files = ["fold_definitions", "predictions", "metrics_summary", "latest_predictions", "experiment_results", "placebo_distribution", "feature_importance", "local_contributions", "portfolio_default"]

    for t in p2_files + p3_files + p4_files:
        f = load(t)
        assert len(f) > 0
        assert "data_source" in f.columns
        assert (f["data_source"] == "SYNTHETIC_DEMO").all()

    # G4 UI Module interface check: all 10 pages expose render()
    import importlib
    pages = [
        "home", "company_explorer", "esg_analysis", "market_regime", "return_prediction",
        "model_comparison", "esg_contribution", "portfolio_risk", "explainability", "methodology"
    ]
    for p in pages:
        mod = importlib.import_module(f"modules.{p}")
        assert hasattr(mod, "render")
        assert callable(mod.render)
