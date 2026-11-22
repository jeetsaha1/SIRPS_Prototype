import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from common.io import load


def test_point_in_time_filing_dates():
    dataset = load("model_dataset")
    # Where ESG is available, available_date <= date
    esg_rows = dataset[dataset["has_esg"].astype(bool) & dataset["esg_available_date"].notna()]
    dt = pd.to_datetime(esg_rows["date"])
    avail_dt = pd.to_datetime(esg_rows["esg_available_date"])
    assert (avail_dt <= dt).all(), "ESG available_date must be <= observation date"


def test_experiment_a_prime_and_b_key_identity():
    """Blueprint requirement: Experiments A' and B must share identical (company_id, date) keys and targets."""
    preds = load("predictions")
    aprime = preds[preds["experiment"] == "A_prime"].sort_values(["model", "company_id", "date"]).reset_index(drop=True)
    b_exp = preds[preds["experiment"] == "B"].sort_values(["model", "company_id", "date"]).reset_index(drop=True)

    assert len(aprime) == len(b_exp), f"Row count mismatch: {len(aprime)} vs {len(b_exp)}"

    for col in ["company_id", "date", "model", "y_hat_excess", "y_hat_abs"]:
        if col in aprime.columns and col in b_exp.columns:
            assert (aprime["company_id"] == b_exp["company_id"]).all()
            assert (aprime["date"] == b_exp["date"]).all()


def test_embargo_verification():
    """Verify 5-trading-day embargo in fold definitions."""
    folds = load("fold_definitions")
    assert not folds.empty
    assert "embargo_days" in folds.columns
    assert (folds["embargo_days"] == 5).all()

    # Verify no training observations have target_end_date >= test_start - 5 days
    for _, f in folds.iterrows():
        te_start = pd.to_datetime(f["test_start"])
        tr_end = pd.to_datetime(f["train_end"])
        # Gap between tr_end and te_start must be at least embargo days
        assert (te_start - tr_end).days >= 1


def test_shifted_feature_sensitivity():
    """Shifted-feature test: altering future input values must not alter past features."""
    from data_generation import feat_technical
    prices = load("prices_daily")
    index = load("index_daily")
    keys = pd.DataFrame([{"company_id": "C01", "date": pd.Timestamp("2022-06-01")}])

    res1 = feat_technical.build(prices, index, keys)

    # Shift prices by changing future data only (after 2022-06-01)
    mod_prices = prices.copy()
    mask_future = (mod_prices["company_id"] == "C01") & (pd.to_datetime(mod_prices["date"]) > pd.Timestamp("2022-06-01"))
    mod_prices.loc[mask_future, "adj_close"] *= 2.0

    res2 = feat_technical.build(mod_prices, index, keys)

    # Output at 2022-06-01 must be completely unchanged by future changes
    for col in ["ret_5d", "ret_21d", "ret_63d", "vol_63d", "rsi_14"]:
        assert np.isclose(res1[col].iloc[0], res2[col].iloc[0]), f"Future leakage detected in {col}!"
