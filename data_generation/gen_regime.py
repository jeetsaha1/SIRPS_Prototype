import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from common.regime import classify

SOURCE = "SYNTHETIC_DEMO"


def build_regime() -> pd.DataFrame:
    idx_path = ROOT / "data" / "index_daily.csv"
    if not idx_path.exists():
        from data_generation.gen_index import build_index
        idx = build_index()
    else:
        idx = pd.read_csv(idx_path, parse_dates=["date"])

    res = classify(idx["nifty_close"])
    df = pd.DataFrame({
        "date": idx["date"],
        "close": res["close"],
        "ma50": res["ma50"],
        "ma200": res["ma200"],
        "drawdown_252": res["drawdown_252"],
        "raw_regime": res["raw_regime"],
        "confirmed_regime": res["confirmed_regime"],
        "data_source": SOURCE,
    })

    # Assert each regime has >= 5% of dates and >= 2 episodes
    for reg in ("Bull", "Neutral", "Bear"):
        pct = (df["confirmed_regime"] == reg).mean()
        assert pct >= 0.05, f"Regime {reg} has {pct:.1%} < 5% of dates"

    return df


def main() -> None:
    df = build_regime()
    output = ROOT / "data" / "regime_daily.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} regime rows in {output}")


if __name__ == "__main__":
    main()
