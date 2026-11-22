import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

SOURCE = "SYNTHETIC_DEMO"


def get_folds() -> pd.DataFrame:
    """Creates 3 expanding walk-forward folds:
    Fold 1: train 2021-01-01 to 2022-12-31, test 2023-01-01 to 2023-12-31
    Fold 2: train 2021-01-01 to 2023-12-31, test 2024-01-01 to 2024-12-31
    Fold 3: train 2021-01-01 to 2024-12-31, test 2025-01-01 to 2025-12-31
    Purge gap: 63 trading days; embargo: 5 trading days.
    """
    rows = []
    experiments = ["A", "A_prime", "B", "C", "P"]
    fold_specs = [
        (1, "2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        (2, "2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        (3, "2021-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ]

    for exp in experiments:
        for f_num, tr_s, tr_e, te_s, te_e in fold_specs:
            rows.append({
                "experiment": exp,
                "fold": f_num,
                "train_start": pd.Timestamp(tr_s),
                "train_end": pd.Timestamp(tr_e),
                "test_start": pd.Timestamp(te_s),
                "test_end": pd.Timestamp(te_e),
                "purged_rows": 63,
                "embargo_days": 5,
                "data_source": SOURCE,
            })

    return pd.DataFrame(rows)


def main():
    df = get_folds()
    output = ROOT / "data" / "fold_definitions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} fold definitions in {output}")


if __name__ == "__main__":
    main()
