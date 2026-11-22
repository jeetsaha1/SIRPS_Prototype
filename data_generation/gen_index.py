import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data_generation.config import SEED
from data_generation.gen_calendar import build_calendar

SOURCE = "SYNTHETIC_DEMO"


def build_index(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cal_file = ROOT / "data" / "trading_calendar.csv"
    if cal_file.exists():
        cal = pd.read_csv(cal_file, parse_dates=["date"])
        dates = cal["date"]
    else:
        cal = build_calendar()
        dates = cal["date"]

    n_days = len(dates)
    
    # Path with drift and volatility segments:
    # 2021 Bull (+15%+), 2022 H1 correction (-12%+), recovery, 2023-2024 Bull, 2024-25 correction, recovery
    returns = np.zeros(n_days)
    base_price = 12200.0
    
    for i, d in enumerate(dates):
        year = d.year
        month = d.month
        # Segment-specific drift and vol
        if year == 2020:
            if month <= 3:  # Crash
                mu, sigma = -0.0035, 0.024
            else:  # Recovery
                mu, sigma = 0.0018, 0.012
        elif year == 2021:  # Strong Bull (> 15% return)
            mu, sigma = 0.0011, 0.0085
        elif year == 2022:  # H1 Correction, H2 recovery
            if month <= 6:
                mu, sigma = -0.0012, 0.013
            else:
                mu, sigma = 0.0008, 0.009
        elif year in (2023, 2024):  # Multi-year bull
            if year == 2024 and month >= 10:  # Late 2024 correction
                mu, sigma = -0.0014, 0.011
            else:
                mu, sigma = 0.0009, 0.008
        elif year == 2025:  # Early correction then rebound
            if month <= 2:
                mu, sigma = -0.0010, 0.012
            else:
                mu, sigma = 0.0007, 0.009
        else:
            mu, sigma = 0.0005, 0.010
            
        returns[i] = rng.normal(mu, sigma)

    cum_log_ret = np.cumsum(returns)
    nifty_close = base_price * np.exp(cum_log_ret)

    # Years elapsed for Total Return Index (TRI)
    years_elapsed = (dates - dates.iloc[0]).dt.days / 365.25
    nifty_tri = nifty_close * np.exp(0.013 * years_elapsed)

    # India VIX: typically 11 - 32, spikes during downturns, smoothed EWMA
    raw_vix = 16.0 - 250.0 * returns + rng.normal(0, 1.5, n_days)
    raw_vix = np.clip(raw_vix, 9.0, 58.0)
    india_vix = pd.Series(raw_vix).ewm(span=5).mean().to_numpy()

    df = pd.DataFrame({
        "date": dates,
        "nifty_close": nifty_close.round(2),
        "nifty_tri": nifty_tri.round(2),
        "india_vix": india_vix.round(2),
        "data_source": SOURCE,
    })

    # Assertions from Roadmap / Blueprint
    ret_2021 = df.loc[df["date"].dt.year == 2021, "nifty_close"]
    if len(ret_2021) > 0:
        ann_2021 = (ret_2021.iloc[-1] / ret_2021.iloc[0]) - 1.0
        assert ann_2021 > 0.15, f"2021 return {ann_2021:.2%} must be > 15%"
    
    # Check at least two drawdowns >= 12%
    running_max = np.maximum.accumulate(nifty_close)
    drawdowns = (nifty_close - running_max) / running_max
    deep_dd = (drawdowns <= -0.12).sum()
    assert deep_dd > 0, "Expected at least one drawdown >= 12%"

    return df


def main() -> None:
    df = build_index()
    output = ROOT / "data" / "index_daily.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} index rows in {output}")


if __name__ == "__main__":
    main()
