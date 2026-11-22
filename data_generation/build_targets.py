import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def compute_targets(prices: pd.DataFrame, index: pd.DataFrame, calendar: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Computes:
    entry_date = calendar[t+1]
    target_end_date = calendar[t+64]
    y_excess = ln(adj[t+64] / adj[t+1]) - ln(nifty_tri[t+64] / nifty_tri[t+1])
    y_abs = ln(adj[t+64] / adj[t+1])
    """
    cal = calendar.sort_values("trading_day_index").reset_index(drop=True)
    date_to_idx = dict(zip(cal["date"], cal["trading_day_index"]))
    idx_to_date = dict(zip(cal["trading_day_index"], cal["date"]))

    prices_map = prices.set_index(["company_id", "date"])["adj_close"].to_dict()
    tri_map = index.set_index("date")["nifty_tri"].to_dict()

    out_rows = []
    for _, row in keys.iterrows():
        cid = row["company_id"]
        dt = row["date"]
        t_idx = date_to_idx.get(dt)

        if t_idx is None or (t_idx + 64) >= len(cal):
            out_rows.append({
                "entry_date": pd.NaT,
                "target_end_date": pd.NaT,
                "y_excess": np.nan,
                "y_abs": np.nan,
            })
            continue

        entry_d = idx_to_date[t_idx + 1]
        exit_d = idx_to_date[t_idx + 64]

        p_entry = prices_map.get((cid, entry_d))
        p_exit = prices_map.get((cid, exit_d))
        tri_entry = tri_map.get(entry_d)
        tri_exit = tri_map.get(exit_d)

        if p_entry and p_exit and tri_entry and tri_exit and p_entry > 0 and tri_entry > 0:
            stock_ret = np.log(p_exit / p_entry)
            index_ret = np.log(tri_exit / tri_entry)
            y_excess = stock_ret - index_ret
            y_abs = stock_ret
        else:
            y_excess = np.nan
            y_abs = np.nan

        out_rows.append({
            "entry_date": entry_d,
            "target_end_date": exit_d,
            "y_excess": y_excess,
            "y_abs": y_abs,
        })

    target_df = pd.DataFrame(out_rows)
    return pd.concat([keys.reset_index(drop=True), target_df], axis=1)
