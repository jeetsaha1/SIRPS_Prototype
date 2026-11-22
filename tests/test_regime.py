import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from common.io import load
from common.regime import classify


def test_regime_classification_rules():
    # Synthetic series for hand-check
    dates = pd.bdate_range("2021-01-01", periods=120)
    # Upward trend
    prices = pd.Series(np.linspace(100, 200, 120), index=dates)
    res = classify(prices, bear_dd=-0.15, confirm_days=10)

    assert "raw_regime" in res.columns
    assert "confirmed_regime" in res.columns
    assert len(res) == 120

    # First confirm_days must start Neutral before switching
    assert (res["confirmed_regime"].iloc[:9] == "Neutral").all()
    # Once MA50 > MA200 and held for 10 days, switches to Bull
    assert res["confirmed_regime"].iloc[70] == "Bull"


def test_regime_truncation_causality():
    """Truncation test: label at date t must be identical whether classifying
    full series or series truncated at t (proves strict causality, zero future leakage).
    """
    index = load("index_daily")
    close = index.set_index("date")["nifty_close"]

    # Classify full series
    full_res = classify(close, bear_dd=-0.15, confirm_days=10)

    # Pick 5 arbitrary test dates across the timeline
    test_indices = [150, 400, 750, 1100, 1400]
    for idx in test_indices:
        trunc_close = close.iloc[: idx + 1]
        trunc_res = classify(trunc_close, bear_dd=-0.15, confirm_days=10)
        assert trunc_res["confirmed_regime"].iloc[-1] == full_res["confirmed_regime"].iloc[idx]


def test_regime_daily_file_properties():
    regime = load("regime_daily")
    assert not regime.empty
    assert len(regime) == 1526

    # Verify each regime has >= 5% of dates
    counts = regime["confirmed_regime"].value_counts(normalize=True)
    for r in ["Bull", "Neutral", "Bear"]:
        assert r in counts
        assert counts[r] >= 0.05, f"Regime {r} has {counts[r]:.1%}, expected >= 5%"

    # Verify each regime has at least 2 episodes
    switches = (regime["confirmed_regime"] != regime["confirmed_regime"].shift(1)).astype(int)
    episodes = regime.groupby(switches.cumsum())["confirmed_regime"].first().value_counts()
    for r in ["Bull", "Neutral", "Bear"]:
        assert episodes.get(r, 0) >= 2, f"Regime {r} has {episodes.get(r, 0)} episodes, expected >= 2"
