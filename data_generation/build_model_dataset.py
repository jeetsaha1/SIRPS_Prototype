import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

import data_generation.feat_esg as feat_esg
import data_generation.feat_fundamental as feat_fundamental
import data_generation.feat_market as feat_market
import data_generation.feat_technical as feat_technical
from data_generation.build_targets import compute_targets

SOURCE = "SYNTHETIC_DEMO"


def build_dataset():
    # Load foundational data
    cal = pd.read_csv(ROOT / "data" / "trading_calendar.csv", parse_dates=["date"])
    idx = pd.read_csv(ROOT / "data" / "index_daily.csv", parse_dates=["date"])
    prices = pd.read_csv(ROOT / "data" / "prices_daily.csv", parse_dates=["date"])
    esg_facts = pd.read_csv(ROOT / "data" / "esg_facts.csv", parse_dates=["filed_timestamp", "available_date"])
    esg_scores = pd.read_csv(ROOT / "data" / "esg_scores.csv")
    funds = pd.read_csv(ROOT / "data" / "fundamentals_pit.csv", parse_dates=["period_end", "filed_timestamp", "available_date"])
    regime = pd.read_csv(ROOT / "data" / "regime_daily.csv", parse_dates=["date"])
    company_master = pd.read_csv(ROOT / "data" / "company_master.csv")

    companies = sorted(company_master["company_id"].unique())
    n_days = len(cal)

    # Warm-up of 252 days; must have t+64 available
    start_idx = 252
    end_idx = n_days - 65

    sampled_indices = list(range(start_idx, end_idx, 5))
    sampled_dates = cal.iloc[sampled_indices]["date"].tolist()

    # Build key grid
    key_rows = []
    for d in sampled_dates:
        for cid in companies:
            key_rows.append({"company_id": cid, "date": d})
    keys = pd.DataFrame(key_rows)

    # 1. Targets
    t_df = compute_targets(prices, idx, cal, keys)

    # 2. Technical features
    tech_df = feat_technical.build(prices, idx, keys)

    # 3. Market features
    mkt_df = feat_market.build(idx, keys)

    # 4. ESG features
    esg_df = feat_esg.build(esg_facts, esg_scores, keys)

    # 5. Fundamental features
    fund_df = feat_fundamental.build(funds, prices, keys)

    # 6. Regime
    regime_map = regime.set_index("date")["confirmed_regime"].to_dict()
    regimes = [regime_map.get(d, "Neutral") for d in keys["date"]]

    # Assemble model_dataset
    dataset = pd.DataFrame({
        "company_id": keys["company_id"],
        "date": keys["date"],
        "entry_date": t_df["entry_date"],
        "target_end_date": t_df["target_end_date"],
        # Technical
        "ret_5d": tech_df["ret_5d"].round(6),
        "ret_21d": tech_df["ret_21d"].round(6),
        "ret_63d": tech_df["ret_63d"].round(6),
        "ret_126d": tech_df["ret_126d"].round(6),
        "mom_12_1": tech_df["mom_12_1"].round(6),
        "price_ma50_ratio": tech_df["price_ma50_ratio"].round(6),
        "price_ma200_ratio": tech_df["price_ma200_ratio"].round(6),
        "vol_63d": tech_df["vol_63d"].round(6),
        "beta_126d": tech_df["beta_126d"].round(6),
        "rsi_14": tech_df["rsi_14"].round(4),
        "volume_ratio": tech_df["volume_ratio"].round(6),
        "maxdd_126d": tech_df["maxdd_126d"].round(6),
        # Market
        "nifty_ret_21d": mkt_df["nifty_ret_21d"].round(6),
        "nifty_ret_63d": mkt_df["nifty_ret_63d"].round(6),
        "nifty_vol_21d": mkt_df["nifty_vol_21d"].round(6),
        "india_vix": mkt_df["india_vix"].round(2),
        "vix_chg_21d": mkt_df["vix_chg_21d"].round(6),
        "nifty_drawdown": mkt_df["nifty_drawdown"].round(6),
        # Regime & Targets
        "regime": regimes,
        "y_excess": t_df["y_excess"],
        "y_abs": t_df["y_abs"],
        # ESG
        "has_esg": esg_df["has_esg"],
        "esg_available_date": esg_df["esg_available_date"],
        "in_A": True,
        "in_matched": esg_df["has_esg"],
        "data_source": SOURCE,
    })

    # Drop any row with NaN target
    dataset = dataset.dropna(subset=["y_excess", "y_abs"]).reset_index(drop=True)

    # Latest features (as of 2025-12-31, one row per company)
    last_date = cal["date"].iloc[-1]
    latest_keys = pd.DataFrame([{"company_id": cid, "date": last_date} for cid in companies])
    l_tech = feat_technical.build(prices, idx, latest_keys)
    l_mkt = feat_market.build(idx, latest_keys)
    l_esg = feat_esg.build(esg_facts, esg_scores, latest_keys)

    latest_features = pd.DataFrame({
        "company_id": latest_keys["company_id"],
        "date": latest_keys["date"],
        "ret_5d": l_tech["ret_5d"].round(6),
        "ret_21d": l_tech["ret_21d"].round(6),
        "ret_63d": l_tech["ret_63d"].round(6),
        "ret_126d": l_tech["ret_126d"].round(6),
        "mom_12_1": l_tech["mom_12_1"].round(6),
        "price_ma50_ratio": l_tech["price_ma50_ratio"].round(6),
        "price_ma200_ratio": l_tech["price_ma200_ratio"].round(6),
        "vol_63d": l_tech["vol_63d"].round(6),
        "beta_126d": l_tech["beta_126d"].round(6),
        "rsi_14": l_tech["rsi_14"].round(4),
        "volume_ratio": l_tech["volume_ratio"].round(6),
        "maxdd_126d": l_tech["maxdd_126d"].round(6),
        "nifty_ret_21d": l_mkt["nifty_ret_21d"].round(6),
        "nifty_ret_63d": l_mkt["nifty_ret_63d"].round(6),
        "nifty_vol_21d": l_mkt["nifty_vol_21d"].round(6),
        "india_vix": l_mkt["india_vix"].round(2),
        "vix_chg_21d": l_mkt["vix_chg_21d"].round(6),
        "nifty_drawdown": l_mkt["nifty_drawdown"].round(6),
        "regime": [regime_map.get(last_date, "Neutral")] * len(companies),
        "has_esg": l_esg["has_esg"],
        "esg_available_date": l_esg["esg_available_date"],
        "data_source": SOURCE,
    })

    return dataset, latest_features


def main():
    dataset, latest_features = build_dataset()
    ds_path = ROOT / "data" / "model_dataset.csv"
    lf_path = ROOT / "data" / "latest_features.csv"
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(ds_path, index=False)
    latest_features.to_csv(lf_path, index=False)
    print(f"Generated {len(dataset)} dataset rows in {ds_path} and {len(latest_features)} rows in {lf_path}")


if __name__ == "__main__":
    main()
