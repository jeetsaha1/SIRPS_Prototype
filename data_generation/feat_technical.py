import numpy as np
import pandas as pd


def build(prices: pd.DataFrame, index: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Builds 12 technical feature columns point-in-time at prediction date `t`.
    
    Columns:
    ret_5d, ret_21d, ret_63d, ret_126d, mom_12_1, price_ma50_ratio, price_ma200_ratio,
    vol_63d, beta_126d, rsi_14, volume_ratio, maxdd_126d.
    """
    prices = prices.sort_values(["company_id", "date"]).copy()
    index = index.sort_values("date").copy()

    # Precompute company-level technical metrics
    comp_feats = {}
    for comp_id, group in prices.groupby("company_id"):
        df = group.set_index("date")
        adj = df["adj_close"]
        vol = df["volume"]

        # Returns
        r5 = adj.pct_change(5)
        r21 = adj.pct_change(21)
        r63 = adj.pct_change(63)
        r126 = adj.pct_change(126)
        r252 = adj.pct_change(252)
        mom_12_1 = (1.0 + r252) / (1.0 + r21) - 1.0

        # MAs
        ma50 = adj.rolling(50, min_periods=20).mean()
        ma200 = adj.rolling(200, min_periods=50).mean()
        p_ma50 = adj / ma50
        p_ma200 = adj / ma200

        # Volatility (63d annualized)
        daily_ret = adj.pct_change()
        vol_63 = daily_ret.rolling(63, min_periods=20).std() * np.sqrt(252)

        # Max drawdown 126d
        roll_max_126 = adj.rolling(126, min_periods=20).max()
        maxdd_126 = (adj - roll_max_126) / roll_max_126

        # Volume ratio (21d avg vs 126d avg)
        vol_21 = vol.rolling(21, min_periods=5).mean()
        vol_126 = vol.rolling(126, min_periods=20).mean()
        vol_ratio = vol_21 / vol_126.replace(0, np.nan)

        # RSI 14
        delta = adj.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=5).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_14 = 100 - (100 / (1 + rs))

        # Beta 126d vs NIFTY
        idx_ret = index.set_index("date")["nifty_close"].pct_change()
        aligned = pd.concat([daily_ret.rename("stock"), idx_ret.rename("index")], axis=1).dropna()
        cov = aligned["stock"].rolling(126, min_periods=30).cov(aligned["index"])
        var = aligned["index"].rolling(126, min_periods=30).var()
        beta_126 = cov / var.replace(0, np.nan)

        feat_df = pd.DataFrame({
            "ret_5d": r5,
            "ret_21d": r21,
            "ret_63d": r63,
            "ret_126d": r126,
            "mom_12_1": mom_12_1,
            "price_ma50_ratio": p_ma50,
            "price_ma200_ratio": p_ma200,
            "vol_63d": vol_63,
            "beta_126d": beta_126,
            "rsi_14": rsi_14,
            "volume_ratio": vol_ratio,
            "maxdd_126d": maxdd_126,
        }, index=df.index)

        comp_feats[comp_id] = feat_df

    # Join onto keys
    out_rows = []
    for _, row in keys.iterrows():
        cid = row["company_id"]
        dt = row["date"]
        if cid in comp_feats and dt in comp_feats[cid].index:
            r = comp_feats[cid].loc[dt].to_dict()
        else:
            r = {k: np.nan for k in [
                "ret_5d", "ret_21d", "ret_63d", "ret_126d", "mom_12_1",
                "price_ma50_ratio", "price_ma200_ratio", "vol_63d", "beta_126d",
                "rsi_14", "volume_ratio", "maxdd_126d",
            ]}
        out_rows.append(r)

    feat_out = pd.DataFrame(out_rows)
    result = pd.concat([keys.reset_index(drop=True), feat_out], axis=1)
    return result
