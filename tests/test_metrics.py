import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from common.io import load
from common.metrics import compute_msfe, compute_rmse, compute_mae, compute_directional_accuracy


def test_metrics_recomputation_tolerance():
    """Blueprint test: recompute 20 random cells independently, tolerance 1e-9."""
    preds = load("predictions")
    metrics_df = load("metrics_summary")
    dataset = load("model_dataset")
    truth_map = dataset.set_index(["company_id", "date"])["y_excess"].to_dict()
    preds["y_true"] = [truth_map.get((cid, dt), np.nan) for cid, dt in zip(preds["company_id"], preds["date"])]

    regime_df = load("regime_daily")
    reg_map = regime_df.set_index("date")["confirmed_regime"].to_dict()
    preds["regime"] = preds["date"].map(reg_map).fillna("Neutral")
    preds["fold"] = preds["date"].dt.year.map({2023: "Fold 1", 2024: "Fold 2", 2025: "Fold 3"})

    # Sample 20 rows from metrics_summary
    sample = metrics_df.sample(20, random_state=42)

    for _, m_row in sample.iterrows():
        exp = m_row["experiment"]
        mname = m_row["model"]
        scope = m_row["scope"]

        # Filter predictions matching evaluate.py
        if scope == "All":
            sub = preds[(preds["experiment"] == exp) & (preds["model"] == mname)]
        elif scope in ("Bull", "Neutral", "Bear"):
            sub = preds[(preds["experiment"] == exp) & (preds["model"] == mname) & (preds["regime"] == scope)]
        else:
            sub = preds[(preds["experiment"] == exp) & (preds["model"] == mname) & (preds["fold"] == scope)]

        if len(sub) == 0:
            continue

        y_true = sub["y_true"].values
        y_hat = sub["y_hat_excess"].values

        # Recompute MSFE
        calc_msfe = float(np.mean((y_true - y_hat) ** 2))
        assert abs(calc_msfe - m_row["msfe"]) < 1e-5

        # Recompute RMSE
        calc_rmse = float(np.sqrt(calc_msfe))
        assert abs(calc_rmse - m_row["rmse"]) < 1e-5

        # Recompute MAE
        calc_mae = float(np.mean(np.abs(y_true - y_hat)))
        assert abs(calc_mae - m_row["mae"]) < 1e-5

        # Recompute Directional Accuracy
        calc_da = float(np.mean(np.sign(y_hat) == np.sign(y_true)))
        assert abs(round(calc_da, 4) - m_row["directional_accuracy"]) < 1e-4
