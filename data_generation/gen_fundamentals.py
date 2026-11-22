import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data_generation.config import SEED
from data_generation.latent import seeded_profile

SOURCE = "SYNTHETIC_DEMO"


def build_fundamentals(seed: int = SEED) -> pd.DataFrame:
    """Generates 27 quarters per firm (2019-03-31 to 2025-09-30).
    
    Item set:
    revenue, net_profit, total_assets, total_debt, equity, eps, operating_cash_flow.
    Filed 38-45 days after period end, available next trading day.
    """
    rng = np.random.default_rng(seed)

    cm_path = ROOT / "data" / "company_master.csv"
    if not cm_path.exists():
        from data_generation.make_company_master import build_company_master
        company_master = build_company_master()
    else:
        company_master = pd.read_csv(cm_path)

    cal_path = ROOT / "data" / "trading_calendar.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path, parse_dates=["date"])
        trading_days = cal["date"].tolist()
    else:
        trading_days = pd.bdate_range("2019-01-01", "2026-06-30").tolist()

    quarter_ends = pd.date_range("2019-03-31", "2025-09-30", freq="QE")
    rows = []

    for comp_idx, comp_id in enumerate(company_master["company_id"], start=1):
        prof = seeded_profile(comp_id, 2023, seed)
        quality = prof["quality_latent"]
        base_revenue = 800.0 + comp_idx * 450.0  # Rs Cr

        # Shares outstanding (Cr shares): e.g. 10 to 50 Cr shares
        shares_out = 15.0 + comp_idx * 3.5

        for q_idx, q_end in enumerate(quarter_ends):
            # Growth with quality drift + noise
            q_factor = (1.0 + 0.02 * quality + 0.02) ** (q_idx / 4.0)
            rev = base_revenue * q_factor * rng.uniform(0.94, 1.06)
            
            # Net margin: 7% - 18% influenced by quality
            net_margin = np.clip(0.08 + 0.03 * quality + rng.normal(0, 0.015), 0.03, 0.25)
            net_profit = rev * net_margin

            # Operating cash flow: ~ 1.1x to 1.5x net profit
            ocf = net_profit * rng.uniform(1.05, 1.45)

            # Assets, debt, equity: total_assets >= total_debt + 0.5 * equity
            equity = rev * rng.uniform(2.5, 4.0)
            debt = equity * rng.uniform(0.2, 0.8)
            total_assets = debt + equity * rng.uniform(0.8, 1.2)

            eps = net_profit / shares_out

            # Filing timestamp: 38 - 45 days after period end
            filing_lag = int(rng.integers(38, 46))
            filed_dt = q_end + pd.Timedelta(days=filing_lag)

            # Available date: next trading day
            future_tds = [d for d in trading_days if d > filed_dt]
            avail_dt = future_tds[0] if future_tds else filed_dt + pd.Timedelta(days=1)

            item_values = {
                "revenue": round(rev, 2),
                "net_profit": round(net_profit, 2),
                "total_assets": round(total_assets, 2),
                "total_debt": round(debt, 2),
                "equity": round(equity, 2),
                "eps": round(eps, 2),
                "operating_cash_flow": round(ocf, 2),
            }

            for item_name, val in item_values.items():
                rows.append({
                    "company_id": comp_id,
                    "period_end": q_end,
                    "period_type": "quarter",
                    "item": item_name,
                    "value": val,
                    "basis": "consolidated",
                    "filed_timestamp": filed_dt,
                    "available_date": avail_dt,
                    "data_source": SOURCE,
                })

    df = pd.DataFrame(rows)
    # Verification: 27 quarters * 10 companies * 7 items = 1890 rows
    assert len(df) == 27 * 10 * 7, f"Expected {27 * 10 * 7} rows, got {len(df)}"
    return df


def main() -> None:
    df = build_fundamentals()
    output = ROOT / "data" / "fundamentals_pit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} fundamentals in {output}")


if __name__ == "__main__":
    main()
