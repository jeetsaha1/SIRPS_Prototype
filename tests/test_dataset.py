import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from common.io import load


def test_dataset_schema_and_technical_features():
    dataset = load("model_dataset")
    assert not dataset.empty
    assert len(dataset) >= 2400

    tech_cols = [
        "ret_5d", "ret_21d", "ret_63d", "ret_126d", "mom_12_1", "price_ma50_ratio",
        "price_ma200_ratio", "vol_63d", "beta_126d", "rsi_14", "volume_ratio", "maxdd_126d",
    ]
    for col in tech_cols:
        assert col in dataset.columns
        assert not dataset[col].isna().any(), f"NaN found in technical feature {col}"

    market_cols = ["nifty_ret_21d", "nifty_ret_63d", "nifty_vol_21d", "india_vix", "vix_chg_21d", "nifty_drawdown"]
    for col in market_cols:
        assert col in dataset.columns
        assert not dataset[col].isna().any(), f"NaN found in market feature {col}"


def test_dataset_five_day_sampling():
    dataset = load("model_dataset")
    calendar = load("trading_calendar")
    date_to_idx = dict(zip(pd.to_datetime(calendar["date"]), calendar["trading_day_index"]))

    # Check for company C01
    c01_dates = sorted(pd.to_datetime(dataset[dataset["company_id"] == "C01"]["date"]).unique())
    indices = [date_to_idx[d] for d in c01_dates]
    diffs = np.diff(indices)

    # Every sampled step must be spaced by exactly 5 trading days
    assert (diffs == 5).all(), f"Observations must be spaced by 5 trading days, found: {set(diffs)}"


def test_dataset_target_recomputation_50_rows():
    """Blueprint test: recompute 50 random y_excess rows to 1e-9 tolerance."""
    dataset = load("model_dataset")
    prices = load("prices_daily")
    index = load("index_daily")

    px_map = prices.set_index(["company_id", "date"])["adj_close"].to_dict()
    tri_map = index.set_index("date")["nifty_tri"].to_dict()

    sample = dataset.sample(50, random_state=42)
    for _, row in sample.iterrows():
        cid = row["company_id"]
        d_en = row["entry_date"]
        d_ex = row["target_end_date"]

        p_en = px_map.get((cid, d_en))
        p_ex = px_map.get((cid, d_ex))
        t_en = tri_map.get(d_en)
        t_ex = tri_map.get(d_ex)

        calc_abs = np.log(p_ex / p_en)
        calc_excess = calc_abs - np.log(t_ex / t_en)

        assert abs(calc_excess - row["y_excess"]) < 1e-9, f"Mismatch on {cid} {row['date']}: {calc_excess} vs {row['y_excess']}"
        assert abs(calc_abs - row["y_abs"]) < 1e-9, f"Mismatch on {cid} {row['date']}: {calc_abs} vs {row['y_abs']}"
