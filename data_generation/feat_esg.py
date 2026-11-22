import numpy as np
import pandas as pd


def build(esg_facts: pd.DataFrame, esg_scores: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Builds ESG point-in-time feature columns at prediction date `t`.
    
    A filing is usable at prediction date t only if available_date <= t.
    Returns: has_esg, esg_available_date, esg_total, esg_coverage, etc.
    """
    facts = esg_facts.copy()
    facts["available_date"] = pd.to_datetime(facts["available_date"])

    scores = esg_scores.copy()
    # Merge available_date from facts (one per company-fiscal_year)
    avail_map = facts.groupby(["company_id", "fiscal_year"])["available_date"].min().reset_index()
    scores = pd.merge(scores, avail_map, on=["company_id", "fiscal_year"], how="left")

    out_rows = []
    for _, row in keys.iterrows():
        cid = row["company_id"]
        dt = pd.to_datetime(row["date"])

        # Eligible filings up to dt
        avail_scores = scores[(scores["company_id"] == cid) & (scores["available_date"] <= dt)].sort_values("fiscal_year")

        if len(avail_scores) > 0:
            latest = avail_scores.iloc[-1]
            out_rows.append({
                "has_esg": True,
                "esg_available_date": latest["available_date"],
                "esg_total": latest["esg_total"],
                "esg_coverage": latest["esg_coverage"],
                "esg_environment": latest.get("esg_environment", np.nan),
                "esg_social": latest.get("esg_social", np.nan),
                "esg_governance": latest.get("esg_governance", np.nan),
            })
        else:
            out_rows.append({
                "has_esg": False,
                "esg_available_date": pd.NaT,
                "esg_total": np.nan,
                "esg_coverage": 0.0,
                "esg_environment": np.nan,
                "esg_social": np.nan,
                "esg_governance": np.nan,
            })

    feat_out = pd.DataFrame(out_rows)
    return pd.concat([keys.reset_index(drop=True), feat_out], axis=1)
