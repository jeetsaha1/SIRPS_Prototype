import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from common.io import load
from common.portfolio_engine import simulate_portfolio


def test_portfolio_zero_cost_equals_gross():
    preds = load("predictions")
    prices = load("prices_daily")
    index = load("index_daily")

    # Run with zero cost
    t0 = time.time()
    res_zero = simulate_portfolio(preds, prices, index, cost=0.0, n_random=50)
    elapsed = time.time() - t0

    # Runtime assertion: under 2 seconds
    assert elapsed < 2.0, f"Simulation took {elapsed:.2f}s, expected < 2.0s"

    # Run with standard cost 0.3%
    res_cost = simulate_portfolio(preds, prices, index, cost=0.003, n_random=50)

    # Net wealth with cost must be less than gross wealth (zero cost)
    final_zero = res_zero["daily"]["strategy"].iloc[-1]
    final_cost = res_cost["daily"]["strategy"].iloc[-1]
    assert final_cost < final_zero, "Net return with costs must be strictly less than gross return"


def test_portfolio_cohort_structure_and_benchmarks():
    preds = load("predictions")
    prices = load("prices_daily")
    index = load("index_daily")
    regime = load("regime_daily")
    comp = load("company_master")

    res = simulate_portfolio(preds, prices, index, regime=regime, company_master=comp, k=3, cost=0.003, n_random=50)
    daily = res["daily"]
    metrics = res["metrics"]
    holdings = res["holdings"]
    by_regime = res["by_regime"]

    # Daily columns check
    expected_cols = ["date", "strategy", "nifty_tri", "equal_weight", "random_mean", "random_p05", "random_p95"]
    for col in expected_cols:
        assert col in daily.columns

    # Random top-k percentile integrity
    assert (daily["random_p95"] >= daily["random_p05"]).all()

    # Cohort count check: ~30-40 cohorts over 1,526 days
    assert metrics["cohort_count"] >= 25

    # Holdings check
    assert not holdings.empty
    assert len(holdings) == 3  # k=3
    assert set(holdings["company_id"]).issubset(set(comp["company_id"]))
    assert np.isclose(holdings["weight"].sum(), 1.0, atol=1e-3)

    # By-regime breakdown check
    assert not by_regime.empty
    assert set(by_regime["regime"]).issubset({"Bull", "Neutral", "Bear"})
    assert "evidence_level" in by_regime.columns
