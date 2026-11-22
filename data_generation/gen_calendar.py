import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data_generation.config import END_DATE, SEED, START_DATE

SOURCE = "SYNTHETIC_DEMO"


def build_calendar(start_date: str = START_DATE, end_date: str = END_DATE, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_weekdays = pd.bdate_range(start=start_date, end=end_date)

    # Standard Indian Market Holidays (fixed date holidays)
    holidays = set()
    for year in range(all_weekdays[0].year, all_weekdays[-1].year + 1):
        holidays.add(pd.Timestamp(f"{year}-01-26"))  # Republic Day
        holidays.add(pd.Timestamp(f"{year}-08-15"))  # Independence Day
        holidays.add(pd.Timestamp(f"{year}-10-02"))  # Gandhi Jayanti
        holidays.add(pd.Timestamp(f"{year}-12-25"))  # Christmas

        # 4 seeded random weekdays per year (Diwali, Holi, Eid, etc.)
        year_weekdays = [d for d in all_weekdays if d.year == year and d not in holidays]
        if len(year_weekdays) >= 4:
            chosen = rng.choice(year_weekdays, size=4, replace=False)
            for d in chosen:
                holidays.add(pd.Timestamp(d))

    trading_dates = [d for d in all_weekdays if d not in holidays]
    trading_dates = sorted(trading_dates)

    df = pd.DataFrame({
        "date": trading_dates,
        "trading_day_index": np.arange(len(trading_dates), dtype=int),
        "year": [d.year for d in trading_dates],
        "is_month_end": [d.is_month_end for d in trading_dates],
        "data_source": SOURCE,
    })
    return df


def main() -> None:
    df = build_calendar()
    output = ROOT / "data" / "trading_calendar.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} trading days in {output}")


if __name__ == "__main__":
    main()
