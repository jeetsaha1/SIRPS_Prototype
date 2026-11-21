import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from data_generation.config import SEED
from data_generation.latent import seeded_profile


def build_company_master() -> pd.DataFrame:
    firms = [
        ("C01", "Surya Grid Renewables", "INPY01", "SURYA", "Renewable Energy", "Solar"),
        ("C02", "Aarav Wind Systems", "INPY02", "AARAV", "Renewable Energy", "Wind"),
        ("C03", "Neer Water Solutions", "INPY03", "NEER", "Utilities", "Water"),
        ("C04", "Prithvi Recycling", "INPY04", "PRITHVI", "Industrials", "Waste"),
        ("C05", "Vaayu Grid Technology", "INPY05", "VAAYU", "Industrials", "Grid"),
        ("C06", "Harit Battery Systems", "INPY06", "HARIT", "Industrials", "Battery"),
        ("C07", "Nadi Waste Recovery", "INPY07", "NADI", "Utilities", "Waste"),
        ("C08", "Agni EV Components", "INPY08", "AGNI", "Automotive", "EV"),
        ("C09", "Jal Energy Efficiency", "INPY09", "JALEFF", "Industrials", "Efficiency"),
        ("C10", "Vana Battery Materials", "INPY10", "VANA", "Materials", "Battery"),
    ]
    rows = []
    for company_id, name, isin, symbol, sector, sub_sector in firms:
        profile = seeded_profile(company_id, 2025, SEED)
        evidence = "Synthetic qualifying-revenue and filing evidence; public source citations not present"
        rows.append({
            "company_id": company_id,
            "name": name,
            "ISIN": isin,
            "NSE_symbol": symbol,
            "sector": sector,
            "sub_sector": sub_sector,
            "universe_group": "core",
            "eligibility_evidence": evidence,
            "entry_date": pd.Timestamp("2020-01-01"),
            "exit_date": pd.NaT,
            "data_source": "SYNTHETIC_DEMO",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    frame = build_company_master()
    output = ROOT / "data" / "company_master.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
