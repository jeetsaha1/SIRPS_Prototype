import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from common.schemas import FILES


FAILURES = []


def check(name: str, frame: pd.DataFrame) -> bool:
    spec = FILES[name]
    missing = set(spec["columns"]) - set(frame.columns)
    if missing:
        print(f"FAIL {name}: missing columns {sorted(missing)}")
        FAILURES.append(name)
        return False
    if "data_source" in frame.columns:
        bad = frame["data_source"].ne("SYNTHETIC_DEMO").sum()
        if bad:
            print(f"FAIL {name}: {bad} rows are not SYNTHETIC_DEMO")
            FAILURES.append(name)
            return False
    print(f"PASS {name}")
    return True


def main() -> int:
    for name, spec in FILES.items():
        real = ROOT / spec["folder"] / spec["file"]
        mock = ROOT / "data" / "mock" / spec["file"]
        path = real if real.exists() else mock
        if not path.exists():
            print(f"FAIL {name}: file missing")
            FAILURES.append(name)
            continue
        frame = pd.read_csv(path)
        check(name, frame)

    company = ROOT / "data" / "company_master.csv"
    if company.exists():
        frame = pd.read_csv(company)
        if len(frame) != 10:
            print("FAIL company_master: expected 10 fictional firms")
            FAILURES.append("company_master_count")
        elif set(frame["company_id"]) != {f"C{i:02d}" for i in range(1, 11)}:
            print("FAIL company_master: expected C01-C10")
            FAILURES.append("company_master_keys")
        else:
            print("PASS company_master_count")

    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
