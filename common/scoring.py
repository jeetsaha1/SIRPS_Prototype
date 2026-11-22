import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

SOURCE = "SYNTHETIC_DEMO"


def score_universe(facts: pd.DataFrame) -> pd.DataFrame:
    """Computes percentile rank ESG scores per fiscal year.
    
    Formula from Roadmap 6.2:
        percentile rank = 100 * (rank - 1) / (valid_firms - 1)
        ties averaged, flipped for lower-is-better indicators.
        A pillar needs >= 60% of its indicators.
        esg_total = equal-weight mean of pillars.
    """
    indicator_pillars = {
        "renewable_energy_share": ("E", False),
        "energy_intensity": ("E", True),
        "emissions_intensity": ("E", True),
        "water_intensity": ("E", True),
        "waste_recovery_rate": ("E", False),
        "lost_time_injury_frequency": ("S", True),
        "women_in_workforce": ("S", False),
        "independent_directors": ("G", False),
    }

    score_rows = []
    fiscal_years = sorted(facts["fiscal_year"].unique())
    companies = sorted(facts["company_id"].unique())

    for fy in fiscal_years:
        fy_facts = facts[facts["fiscal_year"] == fy].copy()

        # Compute percentile ranks for each indicator
        ranks_by_ind = {}
        for ind_id, (pillar, lower_better) in indicator_pillars.items():
            ind_data = fy_facts[fy_facts["indicator_id"] == ind_id]
            valid = ind_data[ind_data["status"] == "value"].dropna(subset=["value"])
            n_valid = len(valid)

            if n_valid > 1:
                # Rank values
                ascending = lower_better  # Lower is better -> lowest raw value gets highest score
                ranked = valid["value"].rank(ascending=not lower_better, method="average")
                pct = 100.0 * (ranked - 1.0) / (n_valid - 1.0)
                ranks_by_ind[ind_id] = pd.Series(pct.values, index=valid["company_id"])
            else:
                ranks_by_ind[ind_id] = pd.Series(dtype=float)

        for comp_id in companies:
            e_scores = []
            s_scores = []
            g_scores = []

            for ind_id, (pillar, _) in indicator_pillars.items():
                if ind_id in ranks_by_ind and comp_id in ranks_by_ind[ind_id].index:
                    score_val = ranks_by_ind[ind_id][comp_id]
                    if pillar == "E":
                        e_scores.append(score_val)
                    elif pillar == "S":
                        s_scores.append(score_val)
                    elif pillar == "G":
                        g_scores.append(score_val)

            # Pillar score requires >= 60% indicator reporting
            e_score = np.mean(e_scores) if len(e_scores) >= 3 else np.nan  # 3 of 5
            s_score = np.mean(s_scores) if len(s_scores) >= 1 else np.nan  # 1 of 2
            g_score = np.mean(g_scores) if len(g_scores) >= 1 else np.nan  # 1 of 1

            total_valid_indicators = len(e_scores) + len(s_scores) + len(g_scores)
            coverage = total_valid_indicators / len(indicator_pillars)

            if pd.notna(e_score) and pd.notna(s_score) and pd.notna(g_score):
                esg_total = (e_score + s_score + g_score) / 3.0
            else:
                esg_total = np.nan

            score_rows.append({
                "company_id": comp_id,
                "fiscal_year": fy,
                "esg_environment": round(float(e_score), 2) if pd.notna(e_score) else np.nan,
                "esg_social": round(float(s_score), 2) if pd.notna(s_score) else np.nan,
                "esg_governance": round(float(g_score), 2) if pd.notna(g_score) else np.nan,
                "esg_total": round(float(esg_total), 2) if pd.notna(esg_total) else np.nan,
                "esg_coverage": round(float(coverage), 2),
                "data_source": SOURCE,
            })

    return pd.DataFrame(score_rows)


def main() -> None:
    facts_path = ROOT / "data" / "esg_facts.csv"
    if not facts_path.exists():
        from data_generation.gen_esg import build_esg_facts
        facts = build_esg_facts()
    else:
        facts = pd.read_csv(facts_path)

    scores_df = score_universe(facts)
    output = ROOT / "data" / "esg_scores.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output, index=False)
    print(f"Generated {len(scores_df)} ESG scores in {output}")


if __name__ == "__main__":
    main()
