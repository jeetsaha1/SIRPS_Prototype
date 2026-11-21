import os
from pathlib import Path

import pandas as pd
import streamlit as st

from common import schemas

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def load(name: str) -> pd.DataFrame:
    if name not in schemas.FILES:
        raise KeyError(f"Unknown schema name: {name}")
    spec = schemas.FILES[name]
    real = ROOT / spec["folder"] / spec["file"]
    mock = ROOT / "data" / "mock" / spec["file"]
    use_mock = os.environ.get("SIRPS_USE_MOCK") == "1" or not real.exists()
    source = mock if use_mock else real
    df = pd.read_csv(source, parse_dates=spec.get("dates", []))
    missing = set(spec["columns"]) - set(df.columns)
    if missing:
        raise AssertionError(f"{name}: missing columns {missing}")
    if "data_source" in df.columns and not df["data_source"].eq("SYNTHETIC_DEMO").all():
        raise AssertionError(f"{name}: bad data_source")
    df.attrs["using_mock"] = use_mock
    df.attrs["source_path"] = str(source)
    return df


def is_using_mock(name: str) -> bool:
    df = load(name)
    return bool(df.attrs.get("using_mock", False))
