import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data_generation.config import (
    PLANTED_ESG,
    PLANTED_MOMENTUM,
    PLANTED_QUALITY,
    SEED,
)
from data_generation.latent import seeded_profile

SOURCE = "SYNTHETIC_DEMO"


def build_prices_and_actions(seed: int = SEED):
    rng = np.random.default_rng(seed)

    cal_path = ROOT / "data" / "trading_calendar.csv"
    if not cal_path.exists():
        from data_generation.gen_calendar import build_calendar
        cal = build_calendar()
    else:
        cal = pd.read_csv(cal_path, parse_dates=["date"])

    cm_path = ROOT / "data" / "company_master.csv"
    if not cm_path.exists():
        from data_generation.make_company_master import build_company_master
        company_master = build_company_master()
    else:
        company_master = pd.read_csv(cm_path)

    dates = cal["date"]
    n_dates = len(dates)

    # Base price levels for C01 - C10
    base_prices = {
        "C01": 412.50, "C02": 285.00, "C03": 609.80, "C04": 215.40, "C05": 540.00,
        "C06": 780.00, "C07": 320.00, "C08": 495.00, "C09": 185.00, "C10": 260.00,
    }

    # Generate corporate actions (2 split/bonus events per firm, plus annual dividends)
    action_rows = []
    split_factors_by_company = {}

    for i, company_id in enumerate(company_master["company_id"], start=1):
        # 1 stock split (e.g. 2-for-1)
        split_date = dates.iloc[min(500 + i * 55, n_dates - 150)]
        action_rows.append({
            "company_id": company_id,
            "ex_date": split_date,
            "action": "split",
            "factor": 2.0,
            "data_source": SOURCE,
        })
        # 1 bonus issue (e.g. 1-for-1 or 1-for-2)
        bonus_date = dates.iloc[min(900 + i * 40, n_dates - 60)]
        action_rows.append({
            "company_id": company_id,
            "ex_date": bonus_date,
            "action": "bonus",
            "factor": 1.5 if i % 2 == 0 else 2.0,
            "data_source": SOURCE,
        })
        # 1 dividend per year
        for yr_idx, yr in enumerate(range(2020, 2026)):
            div_date = dates.iloc[min(100 + yr_idx * 240 + i * 10, n_dates - 10)]
            action_rows.append({
                "company_id": company_id,
                "ex_date": div_date,
                "action": "dividend",
                "factor": float(np.round(rng.uniform(2.0, 15.0), 2)),
                "data_source": SOURCE,
            })

    actions_df = pd.DataFrame(action_rows).sort_values(["company_id", "ex_date"]).reset_index(drop=True)

    # Price generation
    price_rows = []

    for idx, company_id in enumerate(company_master["company_id"], start=1):
        prof = seeded_profile(company_id, 2023, seed)
        base = base_prices.get(company_id, 350.0)
        beta = prof["beta"]
        idio_vol = prof["idiosyncratic_volatility"]
        base_drift = prof["base_drift"]
        esg_latent = prof["esg_latent"]
        quality_latent = prof["quality_latent"]

        # Cumulative split factors backwards from latest to earliest so current price is aligned
        comp_splits = actions_df[(actions_df["company_id"] == company_id) & (actions_df["action"].isin(["split", "bonus"]))]
        
        # Build split multiplier mapping for each date
        split_multiplier = np.ones(n_dates)
        for _, act in comp_splits.iterrows():
            ex_d = act["ex_date"]
            fac = act["factor"]
            mask = dates < ex_d
            split_multiplier[mask] /= fac  # historic raw close is split-adjusted backwards

        # Daily returns
        daily_drift = base_drift / 252.0
        # Add planted signals:
        daily_momentum_drift = (PLANTED_MOMENTUM * 0.5) / 63.0
        daily_quality_drift = (PLANTED_QUALITY * quality_latent) / 63.0
        daily_esg_drift = (PLANTED_ESG * esg_latent) / 63.0

        daily_idio = rng.normal(0, idio_vol / np.sqrt(252), n_dates)
        daily_ret = daily_drift + daily_momentum_drift + daily_quality_drift + daily_esg_drift + daily_idio

        adj_close_series = base * np.exp(np.cumsum(daily_ret))
        raw_close_series = adj_close_series * split_multiplier

        for j in range(n_dates):
            d = dates.iloc[j]
            close = float(raw_close_series[j])
            adj_close = float(adj_close_series[j])

            # Guarantee OHLC consistency: high >= max(open, close), low <= min(open, close)
            # Intraday fluctuation
            day_vol = rng.uniform(0.008, 0.022)
            open_p = close * rng.uniform(0.995, 1.005)
            high_p = max(open_p, close) * (1.0 + rng.uniform(0.002, day_vol))
            low_p = min(open_p, close) * (1.0 - rng.uniform(0.002, day_vol))

            # Volume: lognormal, higher on bigger moves
            ret_abs = abs(daily_ret[j])
            vol_mean = 13.5 + 15.0 * ret_abs
            volume = int(np.exp(rng.normal(vol_mean, 0.4)))

            price_rows.append({
                "company_id": company_id,
                "date": d,
                "open": round(open_p, 2),
                "high": round(high_p, 2),
                "low": round(low_p, 2),
                "close": round(close, 2),
                "adj_close": round(adj_close, 2),
                "volume": volume,
                "source": "SYNTHETIC_GENERATOR",
                "ingest_date": d,
                "data_source": SOURCE,
            })

    prices_df = pd.DataFrame(price_rows)
    return prices_df, actions_df


def main() -> None:
    prices_df, actions_df = build_prices_and_actions()
    p_out = ROOT / "data" / "prices_daily.csv"
    a_out = ROOT / "data" / "corporate_actions.csv"
    p_out.parent.mkdir(parents=True, exist_ok=True)
    prices_df.to_csv(p_out, index=False)
    actions_df.to_csv(a_out, index=False)
    print(f"Generated {len(prices_df)} prices in {p_out} and {len(actions_df)} corporate actions in {a_out}")


if __name__ == "__main__":
    main()
