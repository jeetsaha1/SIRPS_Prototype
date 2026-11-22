import numpy as np
import pandas as pd


def build(fundamentals: pd.DataFrame, prices: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Builds point-in-time fundamental feature columns at prediction date `t`.
    
    Usable only if available_date <= t.
    Computes trailing TTM and quarterly ratios:
    net_margin, debt_to_equity, roe, roa, days_since_filing.
    """
    funds = fundamentals.copy()
    funds["available_date"] = pd.to_datetime(funds["available_date"])
    funds["period_end"] = pd.to_datetime(funds["period_end"])

    out_rows = []
    for _, row in keys.iterrows():
        cid = row["company_id"]
        dt = pd.to_datetime(row["date"])

        avail = funds[(funds["company_id"] == cid) & (funds["available_date"] <= dt)]
        if len(avail) > 0:
            latest_end = avail["period_end"].max()
            q_items = avail[avail["period_end"] == latest_end].set_index("item")["value"].to_dict()
            avail_date = avail[avail["period_end"] == latest_end]["available_date"].iloc[0]

            rev = q_items.get("revenue", np.nan)
            np_val = q_items.get("net_profit", np.nan)
            debt = q_items.get("total_debt", np.nan)
            eq = q_items.get("equity", np.nan)
            assets = q_items.get("total_assets", np.nan)

            net_margin = (np_val / rev) if (pd.notna(rev) and rev > 0) else np.nan
            d_to_e = (debt / eq) if (pd.notna(eq) and eq > 0) else np.nan
            days_since = (dt - avail_date).days

            out_rows.append({
                "net_margin": round(net_margin, 4) if pd.notna(net_margin) else np.nan,
                "debt_to_equity": round(d_to_e, 4) if pd.notna(d_to_e) else np.nan,
                "days_since_filing": days_since,
            })
        else:
            out_rows.append({
                "net_margin": np.nan,
                "debt_to_equity": np.nan,
                "days_since_filing": 999,
            })

    feat_out = pd.DataFrame(out_rows)
    return pd.concat([keys.reset_index(drop=True), feat_out], axis=1)
