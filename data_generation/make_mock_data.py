import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data_generation.config import (
    END_DATE,
    SEED,
    START_DATE,
)
from data_generation.make_company_master import build_company_master


SOURCE = "SYNTHETIC_DEMO"


def save(name: str, frame: pd.DataFrame) -> None:
    output = ROOT / "data" / "mock" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def build() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(START_DATE, END_DATE)

    company_master = build_company_master()
    save("company_master.csv", company_master)

    calendar_dates = dates[:20]
    calendar = pd.DataFrame({
        "date": calendar_dates,
        "trading_day_index": np.arange(len(calendar_dates)),
        "year": calendar_dates.year,
        "is_month_end": calendar_dates.is_month_end,
        "data_source": SOURCE,
    })
    save("trading_calendar.csv", calendar)

    index_close = 18000 * np.exp(np.cumsum(rng.normal(0.00025, 0.008, len(dates))))
    index = pd.DataFrame({
        "date": dates,
        "nifty_close": index_close,
        "nifty_tri": index_close * 1.08,
        "india_vix": np.clip(rng.normal(18, 4, len(dates)), 9, 60),
        "data_source": SOURCE,
    })
    save("index_daily.csv", index)

    price_rows = []
    action_rows = []
    for i, company in enumerate(company_master["company_id"], start=1):
        base = 400 + i * 90
        adj = base * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 20)))
        for j, date in enumerate(dates[:20]):
            close = float(adj[j])
            open_price = close * rng.uniform(0.99, 1.01)
            high = max(open_price, close) * rng.uniform(1.001, 1.025)
            low = min(open_price, close) * rng.uniform(0.975, 0.999)
            price_rows.append({
                "company_id": company,
                "date": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": close,
                "volume": int(rng.lognormal(14.5, 0.35)),
                "source": "SYNTHETIC_GENERATOR",
                "ingest_date": date,
                "data_source": SOURCE,
            })
    save("prices_daily.csv", pd.DataFrame(price_rows))

    for i, company in enumerate(company_master["company_id"], start=1):
        action_rows.append({
            "company_id": company,
            "ex_date": pd.Timestamp("2023-06-30") + pd.Timedelta(days=(i % 4) * 7),
            "action": "split",
            "factor": 2.0,
            "data_source": SOURCE,
        })
    save("corporate_actions.csv", pd.DataFrame(action_rows))

    indicator_ids = [
        "renewable_energy_share",
        "energy_intensity",
        "emissions_intensity",
        "water_intensity",
        "waste_recovery_rate",
        "lost_time_injury_frequency",
        "women_in_workforce",
        "independent_directors",
    ]
    esg_rows = []
    for i, company in enumerate(company_master["company_id"][:10]):
        for j, indicator in enumerate(indicator_ids):
            if (i + j) % 17 == 0:
                status = "NR"
                value = np.nan
            else:
                status = "value"
                value = rng.uniform(20, 95)
            esg_rows.append({
                "company_id": company,
                "fiscal_year": 2025,
                "indicator_id": indicator,
                "value": value,
                "unit_raw": "%",
                "unit_std": "percent",
                "status": status,
                "source_doc": "SYNTHETIC_BRSR_2025.pdf",
                "page": j + 1,
                "filed_timestamp": pd.Timestamp("2025-07-15"),
                "available_date": pd.Timestamp("2025-07-16"),
                "restated_flag": False,
                "entered_by": "P1_SYNTHETIC",
                "verified_by": "P1_SYNTHETIC",
                "data_source": SOURCE,
            })
    save("esg_facts.csv", pd.DataFrame(esg_rows))

    esg_scores = []
    for i, company in enumerate(company_master["company_id"]):
        environment = rng.uniform(55, 85)
        social = rng.uniform(50, 80)
        governance = rng.uniform(60, 90)
        esg_scores.append({
            "company_id": company,
            "fiscal_year": 2025,
            "esg_environment": environment,
            "esg_social": social,
            "esg_governance": governance,
            "esg_total": (environment + social + governance) / 3,
            "esg_coverage": rng.uniform(0.75, 1.0),
            "data_source": SOURCE,
        })
    save("esg_scores.csv", pd.DataFrame(esg_scores))

    fundamentals = []
    for i, company in enumerate(company_master["company_id"]):
        period_end = pd.Timestamp("2025-06-30")
        items = {
            "revenue": rng.uniform(1000, 9000),
            "net_profit": rng.uniform(80, 900),
            "total_assets": rng.uniform(5000, 30000),
            "total_debt": rng.uniform(500, 6000),
            "equity": rng.uniform(2000, 15000),
            "eps": rng.uniform(5, 80),
            "operating_cash_flow": rng.uniform(200, 1800),
        }
        for item, value in items.items():
            fundamentals.append({
                "company_id": company,
                "period_end": period_end,
                "period_type": "quarter",
                "item": item,
                "value": value,
                "basis": "consolidated",
                "filed_timestamp": pd.Timestamp("2025-08-12"),
                "available_date": pd.Timestamp("2025-08-13"),
                "data_source": SOURCE,
            })
    save("fundamentals_pit.csv", pd.DataFrame(fundamentals))

    regime_frame = pd.DataFrame({
        "date": index["date"][:20],
        "close": index["nifty_close"].iloc[:20].to_numpy(),
        "ma50": index["nifty_close"].iloc[:20].rolling(50, min_periods=1).mean().to_numpy(),
        "ma200": index["nifty_close"].iloc[:20].rolling(200, min_periods=1).mean().to_numpy(),
        "drawdown_252": rng.uniform(-0.25, 0.02, 20),
        "raw_regime": np.resize(["Bull", "Neutral", "Bear"], 20),
        "confirmed_regime": np.resize(["Bull", "Neutral", "Bear"], 20),
        "data_source": SOURCE,
    })
    save("regime_daily.csv", regime_frame)

    model_rows = []
    market_row_count = 800
    for i in range(market_row_count):
        company = f"C{(i % 10) + 1:02d}"
        date = dates[20 + i % (len(dates) - 84)]
        model_rows.append({
            "company_id": company,
            "date": date,
            "entry_date": date,
            "target_end_date": date + pd.Timedelta(days=64),
            "ret_5d": rng.normal(0, 0.03),
            "ret_21d": rng.normal(0, 0.05),
            "ret_63d": rng.normal(0, 0.08),
            "ret_126d": rng.normal(0, 0.12),
            "mom_12_1": rng.normal(0, 0.15),
            "price_ma50_ratio": rng.normal(1, 0.08),
            "price_ma200_ratio": rng.normal(1, 0.12),
            "vol_63d": rng.uniform(0.1, 0.5),
            "beta_126d": rng.uniform(0.5, 1.5),
            "rsi_14": rng.uniform(20, 80),
            "volume_ratio": rng.uniform(0.5, 2.0),
            "maxdd_126d": rng.uniform(-0.4, 0),
            "nifty_ret_21d": rng.normal(0, 0.03),
            "nifty_ret_63d": rng.normal(0, 0.08),
            "nifty_vol_21d": rng.uniform(0.05, 0.25),
            "india_vix": rng.uniform(9, 40),
            "vix_chg_21d": rng.normal(0, 0.1),
            "nifty_drawdown": rng.uniform(-0.3, 0),
            "regime": np.resize(["Bull", "Neutral", "Bear"], 1)[0],
            "y_excess": rng.normal(0, 0.12),
            "y_abs": rng.normal(0.02, 0.16),
            "has_esg": bool(i % 3),
            "esg_available_date": date - pd.Timedelta(days=30) if i % 3 else pd.NaT,
            "in_A": True,
            "in_matched": bool(i % 3),
            "data_source": SOURCE,
        })
    save("model_dataset.csv", pd.DataFrame(model_rows))

    latest_rows = []
    for i, company in enumerate(company_master["company_id"]):
        latest_rows.append({
            "company_id": company,
            "date": dates[-1],
            "ret_5d": rng.normal(0, 0.03),
            "ret_21d": rng.normal(0, 0.05),
            "ret_63d": rng.normal(0, 0.08),
            "ret_126d": rng.normal(0, 0.12),
            "mom_12_1": rng.normal(0, 0.15),
            "price_ma50_ratio": rng.normal(1, 0.08),
            "price_ma200_ratio": rng.normal(1, 0.12),
            "vol_63d": rng.uniform(0.1, 0.5),
            "beta_126d": rng.uniform(0.5, 1.5),
            "rsi_14": rng.uniform(20, 80),
            "volume_ratio": rng.uniform(0.5, 2),
            "maxdd_126d": rng.uniform(-0.4, 0),
            "nifty_ret_21d": rng.normal(0, 0.03),
            "nifty_ret_63d": rng.normal(0, 0.08),
            "nifty_vol_21d": rng.uniform(0.05, 0.25),
            "india_vix": rng.uniform(9, 40),
            "vix_chg_21d": rng.normal(0, 0.1),
            "nifty_drawdown": rng.uniform(-0.3, 0),
            "regime": np.resize(["Bull", "Neutral", "Bear"], 1)[0],
            "has_esg": True,
            "esg_available_date": pd.Timestamp("2025-07-16"),
            "data_source": SOURCE,
        })
    save("latest_features.csv", pd.DataFrame(latest_rows))

    fold_rows = []
    for experiment in ("A", "A_prime", "B", "C", "P"):
        for fold, (test_year, test_end) in enumerate(((2023, "2023-12-31"), (2024, "2024-12-31"), (2025, "2025-12-31")), start=1):
            fold_rows.append({
                "experiment": experiment,
                "fold": fold,
                "train_start": pd.Timestamp("2021-01-01"),
                "train_end": pd.Timestamp(year=test_year - 1, month=12, day=31),
                "test_start": pd.Timestamp(test_year, 1, 1),
                "test_end": pd.Timestamp(test_end),
                "purged_rows": 63,
                "embargo_days": 5,
                "data_source": SOURCE,
            })
    save("fold_definitions.csv", pd.DataFrame(fold_rows))

    prediction_rows = []
    metric_rows = []
    latest_prediction_rows = []
    for i in range(800):
        company = f"C{(i % 10) + 1:02d}"
        date = dates[20 + i % (len(dates) - 84)]
        y_hat = rng.normal(0.001, 0.08)
        model_name = ["Zero", "Expanding Mean", "Momentum Rank", "Factor Ridge", "Ridge", "Random Forest", "XGBoost"][i % 7]
        experiment = ["A", "A_prime", "B"][i % 3]
        prediction_rows.append({
            "run_id": f"mock-run-{SEED}",
            "experiment": experiment,
            "model": model_name,
            "company_id": company,
            "date": date,
            "y_hat_excess": y_hat,
            "y_hat_abs": y_hat + 0.005,
            "lower80": y_hat - 0.12,
            "upper80": y_hat + 0.12,
            "nifty_expected_63d": 0.018,
            "model_version": "mock-v0",
            "data_cutoff": date,
            "data_source": SOURCE,
        })
    save("predictions.csv", pd.DataFrame(prediction_rows))

    for experiment in ("A", "A_prime", "B"):
        for model_name in ("Zero", "Expanding Mean", "Momentum Rank", "Factor Ridge", "Ridge", "Random Forest", "XGBoost"):
            for scope in ("All", "Bull", "Neutral", "Bear", "Fold 1", "Fold 2", "Fold 3"):
                msfe = rng.uniform(0.008, 0.018)
                metric_rows.append({
                    "experiment": experiment,
                    "model": model_name,
                    "scope": scope,
                    "fold": "All" if scope.startswith("Fold") is False else scope,
                    "msfe": msfe,
                    "rmse": np.sqrt(msfe),
                    "mae": rng.uniform(0.07, 0.11),
                    "r2_zero": rng.uniform(-0.05, 0.05),
                    "r2_expanding_mean": rng.uniform(-0.08, 0.04),
                    "rank_ic_mean": rng.uniform(-0.03, 0.08),
                    "rank_ic_ir": rng.uniform(-0.2, 0.5),
                    "directional_accuracy": rng.uniform(0.48, 0.58),
                    "top3_spread": rng.uniform(-0.01, 0.04),
                    "data_source": SOURCE,
                })
    save("metrics_summary.csv", pd.DataFrame(metric_rows))

    for i, company in enumerate(company_master["company_id"]):
        latest_prediction_rows.append({
            "company_id": company,
            "date": dates[-1],
            "experiment": "B",
            "model": "XGBoost",
            "y_hat_excess": rng.normal(0.002, 0.04),
            "y_hat_abs": rng.normal(0.01, 0.05),
            "lower80": rng.normal(-0.12, 0.01),
            "upper80": rng.normal(0.12, 0.01),
            "regime": "Bull",
            "model_version": "mock-v0",
            "data_cutoff": dates[-1],
            "data_source": SOURCE,
        })
    save("latest_predictions.csv", pd.DataFrame(latest_prediction_rows))

    experiment_rows = []
    for scope in ("All", "Bull", "Neutral", "Bear"):
        experiment_rows.append({
            "experiment_a": "A_prime",
            "experiment_b": "B",
            "model": "Ridge",
            "scope": scope,
            "delta_msfe": rng.normal(0.0002, 0.001),
            "ci_low": -0.003,
            "ci_high": 0.003,
            "clark_west_p": rng.uniform(0.05, 0.95),
            "h2_p": rng.uniform(0.1, 0.9),
            "n_dates": 800,
            "n_episodes": 3,
            "inferential_flag": scope == "All" and True,
            "data_source": SOURCE,
        })
    save("experiment_results.csv", pd.DataFrame(experiment_rows))

    placebo_rows = []
    real_delta = 0.0003
    for permutation in range(30):
        placebo_rows.append({
            "permutation": permutation,
            "experiment": "P",
            "model": "Ridge",
            "delta_msfe": rng.normal(real_delta, 0.001),
            "data_source": SOURCE,
        })
    placebo_rows.append({
        "permutation": "real_B",
        "experiment": "B",
        "model": "Ridge",
        "delta_msfe": real_delta,
        "data_source": SOURCE,
    })
    save("placebo_distribution.csv", pd.DataFrame(placebo_rows))

    feature_names = [
        ("mom_12_1", "technical"), ("ret_63d", "technical"),
        ("net_margin", "fundamental"), ("debt_to_equity", "fundamental"),
        ("nifty_vol_21d", "market"), ("nifty_ret_63d", "market"),
        ("esg_total", "ESG"), ("renewable_energy_share", "ESG"),
        ("company_id", "identity"),
    ]
    feature_rows = []
    for experiment in ("A", "A_prime", "B", "C"):
        for model_name in ("Ridge", "Random Forest", "XGBoost"):
            for feature, group in feature_names:
                feature_rows.append({
                    "experiment": experiment,
                    "model": model_name,
                    "feature": feature,
                    "feature_group": group,
                    "importance": rng.uniform(0.01, 0.16),
                    "data_source": SOURCE,
                })
    save("feature_importance.csv", pd.DataFrame(feature_rows))

    local_rows = []
    for i, company in enumerate(company_master["company_id"]):
        base = rng.normal(0, 0.02)
        contributions = {}
        for j, (feature, group) in enumerate(feature_names[:6]):
            contributions[feature] = rng.normal(0, 0.01)
        y_hat = base + sum(contributions.values())
        for feature, contribution in contributions.items():
            group = dict(feature_names)[feature]
            local_rows.append({
                "company_id": company,
                "date": dates[-1],
                "model": "Ridge",
                "feature": feature,
                "feature_group": group,
                "contribution": contribution,
                "base_value": base,
                "y_hat_excess": y_hat,
                "data_source": SOURCE,
            })
    save("local_contributions.csv", pd.DataFrame(local_rows))

    strategy = np.cumprod(1 + rng.normal(0.0002, 0.008, 20))
    nifty = np.cumprod(1 + rng.normal(0.00015, 0.007, 20))
    equal = np.cumprod(1 + rng.normal(0.0001, 0.006, 20))
    portfolio = pd.DataFrame({
        "date": dates[:20],
        "strategy": strategy,
        "nifty_tri": nifty,
        "equal_weight": equal,
        "random_mean": equal,
        "random_p05": equal * 0.95,
        "random_p95": equal * 1.05,
        "data_source": SOURCE,
    })
    save("portfolio_default.csv", portfolio)


if __name__ == "__main__":
    build()
