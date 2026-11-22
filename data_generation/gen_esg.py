import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data_generation.config import ESG_FILING_DAY, ESG_FILING_MONTH, SEED
from data_generation.latent import seeded_profile

SOURCE = "SYNTHETIC_DEMO"

INDICATOR_SPECS = [
    {"id": "renewable_energy_share", "name": "Renewable Energy Share", "pillar": "E", "unit_raw": "%", "unit_std": "percent", "lower_better": False, "range": (15.0, 95.0)},
    {"id": "energy_intensity", "name": "Energy Intensity", "pillar": "E", "unit_raw": "MWh/Cr", "unit_std": "MWh/Cr", "lower_better": True, "range": (5.0, 85.0)},
    {"id": "emissions_intensity", "name": "GHG Emissions Intensity", "pillar": "E", "unit_raw": "tCO2e/Cr", "unit_std": "tCO2e/Cr", "lower_better": True, "range": (10.0, 120.0)},
    {"id": "water_intensity", "name": "Water Consumption Intensity", "pillar": "E", "unit_raw": "kL/Cr", "unit_std": "kL/Cr", "lower_better": True, "range": (20.0, 250.0)},
    {"id": "waste_recovery_rate", "name": "Waste Recovery Rate", "pillar": "E", "unit_raw": "%", "unit_std": "percent", "lower_better": False, "range": (30.0, 98.0)},
    {"id": "lost_time_injury_frequency", "name": "Lost Time Injury Frequency", "pillar": "S", "unit_raw": "rate", "unit_std": "rate", "lower_better": True, "range": (0.0, 2.5)},
    {"id": "women_in_workforce", "name": "Women in Workforce", "pillar": "S", "unit_raw": "%", "unit_std": "percent", "lower_better": False, "range": (10.0, 42.0)},
    {"id": "independent_directors", "name": "Independent Directors Share", "pillar": "G", "unit_raw": "%", "unit_std": "percent", "lower_better": False, "range": (33.3, 66.7)},
]


def build_esg_facts(seed: int = SEED) -> pd.DataFrame:
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
        trading_days = pd.bdate_range("2020-01-01", "2025-12-31").tolist()

    fiscal_years = [2022, 2023, 2024, 2025]
    rows = []

    for fy in fiscal_years:
        # Filing in mid-July
        filing_dt = pd.Timestamp(year=fy, month=ESG_FILING_MONTH, day=ESG_FILING_DAY)
        # Find next available trading day
        future_tds = [d for d in trading_days if d > filing_dt]
        available_dt = future_tds[0] if future_tds else filing_dt + pd.Timedelta(days=1)

        for comp_id in company_master["company_id"]:
            prof = seeded_profile(comp_id, fy, seed)
            esg_latent = prof["esg_latent"]

            for page_no, ind in enumerate(INDICATOR_SPECS, start=1):
                ind_id = ind["id"]
                # Missingness rate: ~6% baseline, ~12% for water and waste
                nr_prob = 0.12 if "water" in ind_id or "waste" in ind_id else 0.06
                is_nr = rng.random() < nr_prob

                if is_nr:
                    status = "NR"
                    val = np.nan
                else:
                    status = "value"
                    min_v, max_v = ind["range"]
                    # Base value modulated by latent ESG score (higher latent -> better performance)
                    norm_score = (esg_latent + 2.5) / 5.0
                    norm_score = np.clip(norm_score, 0.05, 0.95)
                    if ind["lower_better"]:
                        center = max_v - norm_score * (max_v - min_v)
                    else:
                        center = min_v + norm_score * (max_v - min_v)
                    val = center + rng.normal(0, 0.05 * (max_v - min_v))
                    val = np.clip(val, min_v, max_v)
                    if ind["unit_std"] == "percent":
                        val = np.clip(val, 0.0, 100.0)
                    val = round(float(val), 2)

                rows.append({
                    "company_id": comp_id,
                    "fiscal_year": fy,
                    "indicator_id": ind_id,
                    "value": val,
                    "unit_raw": ind["unit_raw"],
                    "unit_std": ind["unit_std"],
                    "status": status,
                    "source_doc": f"BRSR_{comp_id}_FY{fy}.pdf",
                    "page": page_no,
                    "filed_timestamp": filing_dt,
                    "available_date": available_dt,
                    "restated_flag": False,
                    "entered_by": "P2_SYNTHETIC",
                    "verified_by": "P2_SYNTHETIC",
                    "data_source": SOURCE,
                })

    df = pd.DataFrame(rows)
    # Verification: 8 indicators * 4 years * 10 companies = 320 rows
    assert len(df) == 320, f"Expected 320 rows in esg_facts, got {len(df)}"
    return df


def main() -> None:
    df = build_esg_facts()
    output = ROOT / "data" / "esg_facts.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} ESG facts in {output}")


if __name__ == "__main__":
    main()
