import numpy as np
import pandas as pd


def build(index: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Builds 6 market feature columns point-in-time at prediction date `t`.
    
    Columns:
    nifty_ret_21d, nifty_ret_63d, nifty_vol_21d, india_vix, vix_chg_21d, nifty_drawdown.
    """
    idx = index.sort_values("date").set_index("date").copy()
    close = idx["nifty_close"]
    vix = idx["india_vix"]

    r21 = close.pct_change(21)
    r63 = close.pct_change(63)
    daily_ret = close.pct_change()
    vol21 = daily_ret.rolling(21, min_periods=5).std() * np.sqrt(252)

    vix_chg21 = vix.pct_change(21)

    running_max_252 = close.rolling(252, min_periods=20).max()
    nifty_dd = (close - running_max_252) / running_max_252

    mkt_df = pd.DataFrame({
        "nifty_ret_21d": r21,
        "nifty_ret_63d": r63,
        "nifty_vol_21d": vol21,
        "india_vix": vix,
        "vix_chg_21d": vix_chg21,
        "nifty_drawdown": nifty_dd,
    }, index=idx.index)

    out_rows = []
    for _, row in keys.iterrows():
        dt = row["date"]
        if dt in mkt_df.index:
            r = mkt_df.loc[dt].to_dict()
        else:
            r = {k: np.nan for k in ["nifty_ret_21d", "nifty_ret_63d", "nifty_vol_21d", "india_vix", "vix_chg_21d", "nifty_drawdown"]}
        out_rows.append(r)

    feat_out = pd.DataFrame(out_rows)
    return pd.concat([keys.reset_index(drop=True), feat_out], axis=1)
