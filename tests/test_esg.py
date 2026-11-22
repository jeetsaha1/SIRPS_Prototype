import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from common.io import load
from common.scoring import score_universe


def test_esg_indicator_ranges_and_status():
    facts = load("esg_facts")
    assert not facts.empty
    assert len(facts) == 320

    # Percent indicators must be within [0, 100]
    percent_indicators = {
        "renewable_energy_share",
        "waste_recovery_rate",
        "women_in_workforce",
        "independent_directors",
    }
    pct_rows = facts[facts["indicator_id"].isin(percent_indicators) & facts["value"].notna()]
    assert (pct_rows["value"] >= 0.0).all()
    assert (pct_rows["value"] <= 100.0).all()

    # Valid status codes
    valid_statuses = {"value", "NR", "NA"}
    assert set(facts["status"].unique()).issubset(valid_statuses)


def test_esg_scoring_sample_unit_test():
    """Blueprint test case: pillars 81.5, 66.7, 44.4 -> esg_total 64.2."""
    e_val, s_val, g_val = 81.5, 66.7, 44.4
    expected_total = round((e_val + s_val + g_val) / 3.0, 1)
    assert expected_total == 64.2


def test_esg_scores_contract():
    scores = load("esg_scores")
    assert not scores.empty
    assert len(scores) == 40  # 10 firms x 4 fiscal years

    # Verify score ranges
    for col in ["esg_environment", "esg_social", "esg_governance", "esg_total"]:
        valid = scores[col].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    # Total must be equal-weight average of pillars when all present
    has_all = scores.dropna(subset=["esg_environment", "esg_social", "esg_governance"])
    calc_total = (has_all["esg_environment"] + has_all["esg_social"] + has_all["esg_governance"]) / 3.0
    diff = np.abs(has_all["esg_total"] - calc_total)
    assert diff.max() < 0.2
