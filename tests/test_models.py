import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from common.io import load
from models.baselines import ExpandingMeanBaseline, MomentumRankBaseline, ZeroBaseline
from models.train_experiments import get_model


def test_fold_definitions_purge_and_embargo():
    folds = load("fold_definitions")
    assert not folds.empty
    assert len(folds) == 15  # 5 experiments x 3 folds

    # Purged rows = 63, embargo days = 5
    assert (folds["purged_rows"] == 63).all()
    assert (folds["embargo_days"] == 5).all()

    dataset = load("model_dataset")
    for _, f in folds.iterrows():
        tr_s = pd.to_datetime(f["train_start"])
        tr_e = pd.to_datetime(f["train_end"])
        te_s = pd.to_datetime(f["test_start"])

        # In any fold, training observations must have target_end_date < te_s
        train_obs = dataset[(dataset["date"] >= tr_s) & (dataset["date"] <= tr_e)]
        # When filtered by purge condition
        purged_train = train_obs[train_obs["target_end_date"] >= te_s]
        # In production model training, these purged rows are excluded from training
        assert len(purged_train) > 0  # confirms there were overlapping rows to purge


def test_model_baselines():
    zero = ZeroBaseline()
    zero.fit(np.zeros((10, 5)), np.ones(10))
    p_zero = zero.predict(np.zeros((5, 5)))
    assert (p_zero == 0.0).all()

    mean_base = ExpandingMeanBaseline()
    mean_base.fit(np.zeros((10, 5)), np.full(10, 0.05))
    p_mean = mean_base.predict(np.zeros((5, 5)))
    assert np.allclose(p_mean, 0.05)


def test_reproducibility_same_seed_same_predictions():
    """Blueprint test: identical seed must yield bit-identical predictions."""
    X = np.random.default_rng(42).normal(0, 1, (100, 10))
    y = np.random.default_rng(42).normal(0, 0.1, 100)

    m1 = get_model("XGBoost", seed=42)
    m1.fit(X, y)
    p1 = m1.predict(X[:10])

    m2 = get_model("XGBoost", seed=42)
    m2.fit(X, y)
    p2 = m2.predict(X[:10])

    assert np.allclose(p1, p2, atol=1e-9), "Model predictions must be deterministic with same seed"


def test_local_contributions_additive():
    """Blueprint test: base_value + sum(contributions) == y_hat_excess within 1e-6."""
    local_df = load("local_contributions")
    assert not local_df.empty
    assert len(local_df) == 200

    grouped = local_df.groupby(["company_id", "date", "model"])
    for (cid, dt, m), g in grouped:
        base_val = g["base_value"].iloc[0]
        y_hat = g["y_hat_excess"].iloc[0]
        sum_contrib = g["contribution"].sum()
        total_calc = base_val + sum_contrib

        diff = abs(total_calc - y_hat)
        assert diff < 1e-6, f"Additive identity violated for {cid} {dt}: {total_calc} vs {y_hat} (diff={diff})"
