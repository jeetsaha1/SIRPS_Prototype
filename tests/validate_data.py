import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from common.schemas import FILES


FAILURES = []


def fail(check_name: str, detail: str) -> None:
    print(f"FAIL {check_name}: {detail}")
    FAILURES.append(check_name)


def pass_check(check_name: str) -> None:
    print(f"PASS {check_name}")


def check(name: str, frame: pd.DataFrame) -> bool:
    spec = FILES[name]
    missing = set(spec["columns"]) - set(frame.columns)
    if missing:
        fail(name, f"missing columns {sorted(missing)}")
        return False
    if "data_source" in frame.columns:
        bad = frame["data_source"].ne("SYNTHETIC_DEMO").sum()
        if bad:
            fail(name, f"{bad} rows are not SYNTHETIC_DEMO")
            return False
    pass_check(name)
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
            fail("company_master_count", "expected 10 fictional firms")
        elif set(frame["company_id"]) != {f"C{i:02d}" for i in range(1, 11)}:
            fail("company_master_keys", "expected C01-C10")
        else:
            pass_check("company_master_count")
            pass_check("company_master_keys")

    validate_calendar()
    validate_ohlc()
    validate_esg()
    validate_targets()
    validate_predictions()
    validate_point_in_time()

    return 1 if FAILURES else 0


def load_schema(name: str) -> pd.DataFrame:
    spec = FILES[name]
    real = ROOT / spec["folder"] / spec["file"]
    mock = ROOT / "data" / "mock" / spec["file"]
    path = real if real.exists() else mock
    if not path.exists():
        fail(f"{name}_file", "file missing")
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=spec.get("dates", []))


def validate_calendar() -> None:
    frame = load_schema("trading_calendar")
    if frame.empty:
        return
    if frame["date"].duplicated().any():
        fail("calendar_duplicates", "duplicate trading dates")
        return
    if "trading_day_index" in frame and frame["trading_day_index"].duplicated().any():
        fail("calendar_index", "duplicate trading_day_index values")
        return
    pass_check("calendar_keys")


def validate_ohlc() -> None:
    frame = load_schema("prices_daily")
    if frame.empty:
        return
    bad = (
        (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame["low"])
        | (frame["volume"] < 0)
    )
    if bad.any():
        fail("ohlc_rules", f"{int(bad.sum())} invalid OHLC or volume rows")
    else:
        pass_check("ohlc_rules")


def validate_esg() -> None:
    frame = load_schema("esg_facts")
    if frame.empty:
        return
    percent_ids = {
        "renewable_energy_share",
        "waste_recovery_rate",
        "women_in_workforce",
        "independent_directors",
    }
    percent_rows = frame[frame["indicator_id"].isin(percent_ids) & frame["value"].notna()]
    if "value" in frame and percent_rows["value"].between(0, 100).all():
        pass_check("esg_ranges")
    else:
        fail("esg_ranges", "percent indicator outside 0-100")

    if "NR" in set(frame["status"]) or "NA" in set(frame["status"]) or "value" in set(frame["status"]):
        pass_check("esg_missing_codes")
    else:
        fail("esg_missing_codes", "missing status codes/value status absent")


def validate_targets() -> None:
    frame = load_schema("model_dataset")
    if frame.empty:
        return
    required = ["entry_date", "target_end_date", "y_excess", "y_abs"]
    for column in required:
        if column not in frame:
            fail("target_schema", f"missing {column}")
            return
    if frame["y_excess"].isna().any() or frame["y_abs"].isna().any():
        fail("target_values", "NaN target in model_dataset")
    elif (frame["entry_date"] >= frame["target_end_date"]).any():
        fail("target_dates", "entry_date must precede target_end_date")
    else:
        pass_check("target_dates")
        pass_check("target_values")


def validate_predictions() -> None:
    frame = load_schema("predictions")
    if frame.empty:
        return
    bad_keys = frame.duplicated(["run_id", "experiment", "model", "company_id", "date"]).any()
    if bad_keys:
        fail("prediction_keys", "duplicate prediction rows")
    else:
        pass_check("prediction_keys")
    if (frame["lower80"] > frame["upper80"]).any():
        fail("prediction_range", "lower80 exceeds upper80")
    else:
        pass_check("prediction_range")


def validate_point_in_time() -> None:
    frame = load_schema("model_dataset")
    if frame.empty:
        return
    if "esg_available_date" not in frame or "has_esg" not in frame:
        fail("point_in_time_schema", "model dataset lacks ESG point-in-time fields")
        return
    available = frame.loc[frame["has_esg"].astype(bool), ["date", "esg_available_date"]].dropna()
    if not available.empty and (available["esg_available_date"] > available["date"]).any():
        fail("point_in_time_esg", "ESG is available after prediction date")
    else:
        pass_check("point_in_time_esg")


if __name__ == "__main__":
    raise SystemExit(main())
