import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SOURCE = "SYNTHETIC_DEMO"


def evaluate_all():
    preds = pd.read_csv(ROOT / "results" / "predictions.csv", parse_dates=["date"])
    dataset = pd.read_csv(ROOT / "data" / "model_dataset.csv", parse_dates=["date"])
    regime_df = pd.read_csv(ROOT / "data" / "regime_daily.csv", parse_dates=["date"])
    regime_map = regime_df.set_index("date")["confirmed_regime"].to_dict()

    # Merge true y_excess and regime onto predictions
    truth_map = dataset.set_index(["company_id", "date"])["y_excess"].to_dict()
    preds["y_true"] = [truth_map.get((cid, dt), np.nan) for cid, dt in zip(preds["company_id"], preds["date"])]
    preds["regime"] = [regime_map.get(dt, "Neutral") for dt in preds["date"]]
    preds = preds.dropna(subset=["y_true"]).copy()

    # Fold mapping based on test year
    preds["fold"] = preds["date"].dt.year.map({2023: "Fold 1", 2024: "Fold 2", 2025: "Fold 3"})

    metric_rows = []
    experiments = preds["experiment"].unique()
    models = preds["model"].unique()
    scopes = ["All", "Bull", "Neutral", "Bear", "Fold 1", "Fold 2", "Fold 3"]

    for exp in experiments:
        exp_preds = preds[preds["experiment"] == exp]
        for mname in models:
            m_preds = exp_preds[exp_preds["model"] == mname]
            for scope in scopes:
                if scope == "All":
                    subset = m_preds
                elif scope in ("Bull", "Neutral", "Bear"):
                    subset = m_preds[m_preds["regime"] == scope]
                else:  # Fold 1, 2, 3
                    subset = m_preds[m_preds["fold"] == scope]

                if len(subset) == 0:
                    continue

                y_true = subset["y_true"].values
                y_hat = subset["y_hat_excess"].values

                # Error metrics
                diff = y_true - y_hat
                msfe = float(np.mean(diff ** 2))
                rmse = float(np.sqrt(msfe))
                mae = float(np.mean(np.abs(diff)))

                # R2 vs zero baseline
                denom_zero = float(np.mean(y_true ** 2))
                r2_zero = 1.0 - (msfe / denom_zero) if denom_zero > 0 else 0.0

                # R2 vs expanding mean
                var_y = float(np.var(y_true))
                r2_mean = 1.0 - (msfe / var_y) if var_y > 0 else 0.0

                # Directional accuracy
                dir_acc = float(np.mean(np.sign(y_hat) == np.sign(y_true)))

                # Rank-IC per date
                date_ics = []
                for dt, grp in subset.groupby("date"):
                    if len(grp) >= 3 and grp["y_hat_excess"].nunique() > 1:
                        ic, _ = spearmanr(grp["y_hat_excess"], grp["y_true"])
                        if pd.notna(ic):
                            date_ics.append(ic)

                if len(date_ics) > 0:
                    ic_mean = float(np.mean(date_ics))
                    ic_std = float(np.std(date_ics))
                    ic_ir = (ic_mean / ic_std) if ic_std > 0 else 0.0
                else:
                    ic_mean = 0.0
                    ic_ir = 0.0

                # Top-3 spread: average return of top 3 forecasts vs universe average
                spreads = []
                for dt, grp in subset.groupby("date"):
                    if len(grp) >= 3:
                        top3 = grp.sort_values("y_hat_excess", ascending=False).head(3)
                        spreads.append(top3["y_true"].mean() - grp["y_true"].mean())
                top3_spread = float(np.mean(spreads)) if len(spreads) > 0 else 0.0

                metric_rows.append({
                    "experiment": exp,
                    "model": mname,
                    "scope": scope,
                    "fold": scope if scope.startswith("Fold") else "All",
                    "msfe": round(msfe, 6),
                    "rmse": round(rmse, 6),
                    "mae": round(mae, 6),
                    "r2_zero": round(r2_zero, 6),
                    "r2_expanding_mean": round(r2_mean, 6),
                    "rank_ic_mean": round(ic_mean, 4),
                    "rank_ic_ir": round(ic_ir, 4),
                    "directional_accuracy": round(dir_acc, 4),
                    "top3_spread": round(top3_spread, 6),
                    "data_source": SOURCE,
                })

    df = pd.DataFrame(metric_rows)
    return df


def main():
    df = evaluate_all()
    output = ROOT / "results" / "metrics_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} metric summary rows in {output}")


if __name__ == "__main__":
    main()
