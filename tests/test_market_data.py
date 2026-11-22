import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from common.io import load


def test_ohlc_rules():
    prices = load("prices_daily")
    assert not prices.empty
    assert (prices["high"] >= prices[["open", "close"]].max(axis=1)).all()
    assert (prices["low"] <= prices[["open", "close"]].min(axis=1)).all()
    assert (prices["high"] >= prices["low"]).all()
    assert (prices["volume"] >= 0).all()


def test_calendar_integrity():
    calendar = load("trading_calendar")
    assert not calendar.empty
    assert not calendar["date"].duplicated().any()
    assert not calendar["trading_day_index"].duplicated().any()

    # Verify no weekends
    dates = pd.to_datetime(calendar["date"])
    assert (dates.dt.dayofweek < 5).all()

    # Fixed Indian holidays
    for year in range(2020, 2026):
        assert pd.Timestamp(f"{year}-01-26") not in set(dates)
        assert pd.Timestamp(f"{year}-08-15") not in set(dates)
        assert pd.Timestamp(f"{year}-10-02") not in set(dates)
        assert pd.Timestamp(f"{year}-12-25") not in set(dates)


def test_nifty_segments_and_drawdowns():
    index = load("index_daily")
    assert not index.empty
    dates = pd.to_datetime(index["date"])
    close = index["nifty_close"].values

    # 2021 bull return > +15%
    mask_2021 = dates.dt.year == 2021
    ret_2021 = (close[mask_2021][-1] / close[mask_2021][0]) - 1.0
    assert ret_2021 > 0.15, f"2021 return {ret_2021:.2%} should exceed 15%"

    # At least two distinct drawdowns >= 12% (in 2022 and 2024-2025)
    running_max = np.maximum.accumulate(close)
    drawdowns = (close - running_max) / running_max

    dd_2022 = drawdowns[dates.dt.year == 2022]
    assert np.min(dd_2022) <= -0.12, "Expected >= 12% drawdown in 2022"

    dd_2024_2025 = drawdowns[dates.dt.year.isin([2024, 2025])]
    assert np.min(dd_2024_2025) <= -0.12, "Expected >= 12% drawdown in 2024-2025"


def test_nifty_tri_relationship():
    index = load("index_daily")
    dates = pd.to_datetime(index["date"])
    years = (dates - dates.iloc[0]).dt.days / 365.25
    expected_tri = index["nifty_close"] * np.exp(0.013 * years)
    diff = np.abs(index["nifty_tri"] - expected_tri)
    assert diff.max() < 0.5, "NIFTY TRI must track close * exp(0.013 * years)"


def test_india_vix_range():
    index = load("index_daily")
    assert (index["india_vix"] >= 9.0).all()
    assert (index["india_vix"] <= 60.0).all()
