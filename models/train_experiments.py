import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from data_generation.config import SEED
from models.baselines import ExpandingMeanBaseline, MomentumRankBaseline, ZeroBaseline

SOURCE = "SYNTHETIC_DEMO"
RUN_ID = f"sirps-v1-{SEED}"

TECH_COLS = [
    "ret_5d", "ret_21d", "ret_63d", "ret_126d", "mom_12_1", "price_ma50_ratio",
    "price_ma200_ratio", "vol_63d", "beta_126d", "rsi_14", "volume_ratio", "maxdd_126d",
]
MKT_COLS = ["nifty_ret_21d", "nifty_ret_63d", "nifty_vol_21d", "india_vix", "vix_chg_21d", "nifty_drawdown"]
ESG_COLS = ["esg_total", "esg_coverage"]


def get_feature_columns(experiment: str) -> list:
    if experiment in ("A", "A_prime"):
        return TECH_COLS + MKT_COLS
    elif experiment in ("B", "C", "P"):
        return TECH_COLS + MKT_COLS + ESG_COLS
    return TECH_COLS + MKT_COLS


def get_model(model_name: str, seed: int = SEED):
    """Fixed hyperparameters conforming to Blueprint Section 6.4."""
    if model_name == "Zero":
        return ZeroBaseline()
    elif model_name == "Expanding Mean":
        return ExpandingMeanBaseline()
    elif model_name == "Momentum Rank":
        return MomentumRankBaseline()
    elif model_name == "Factor Ridge":
        return Ridge(alpha=50.0, random_state=seed)
    elif model_name == "Ridge":
        return Ridge(alpha=50.0, random_state=seed)
    elif model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=50,
            max_features=0.5,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "XGBoost":
        return XGBRegressor(
            max_depth=2,
            learning_rate=0.03,
            n_estimators=200,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=10.0,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def standardize_cross_sectionally(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Standardizes features per date cross-sectionally across firms:
    (X - mean(t)) / (std(t) + 1e-8).
    """
    out = df.copy()
    for col in feat_cols:
        if col in out.columns:
            # Group by date and standardize
            grouped = out.groupby("date")[col]
            mean = grouped.transform("mean")
            std = grouped.transform("std").fillna(1.0).replace(0.0, 1.0)
            out[col] = (out[col] - mean) / (std + 1e-8)
    return out


def run_training_and_predictions():
    dataset_path = ROOT / "data" / "model_dataset.csv"
    if not dataset_path.exists():
        from data_generation.build_model_dataset import build_dataset
        dataset, latest_features = build_dataset()
    else:
        dataset = pd.read_csv(dataset_path, parse_dates=["date", "entry_date", "target_end_date"])
        latest_features = pd.read_csv(ROOT / "data" / "latest_features.csv", parse_dates=["date"])

    # Load ESG scores into dataset if esg_total missing
    if "esg_total" not in dataset.columns:
        scores = pd.read_csv(ROOT / "data" / "esg_scores.csv")
        facts = pd.read_csv(ROOT / "data" / "esg_facts.csv", parse_dates=["available_date"])
        avail_map = facts.groupby(["company_id", "fiscal_year"])["available_date"].min().reset_index()
        scores = pd.merge(scores, avail_map, on=["company_id", "fiscal_year"], how="left")
        
        esg_tot = []
        esg_cov = []
        for _, r in dataset.iterrows():
            cid = r["company_id"]
            dt = r["date"]
            avail = scores[(scores["company_id"] == cid) & (scores["available_date"] <= dt)].sort_values("fiscal_year")
            if len(avail) > 0:
                esg_tot.append(avail.iloc[-1]["esg_total"])
                esg_cov.append(avail.iloc[-1]["esg_coverage"])
            else:
                esg_tot.append(50.0)  # Neutral default for unfiled period
                esg_cov.append(0.0)
        dataset["esg_total"] = esg_tot
        dataset["esg_coverage"] = esg_cov

    # Folds
    folds_path = ROOT / "data" / "fold_definitions.csv"
    if not folds_path.exists():
        from models.splits import get_folds
        folds_df = get_folds()
    else:
        folds_df = pd.read_csv(folds_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    models_list = ["Zero", "Expanding Mean", "Momentum Rank", "Factor Ridge", "Ridge", "Random Forest", "XGBoost"]
    experiments = ["A", "A_prime", "B"]

    prediction_rows = []
    saved_models = {}

    # Sort dataset chronologically
    dataset = dataset.sort_values("date").reset_index(drop=True)

    for exp in experiments:
        exp_data = dataset.copy()
        if exp in ("A_prime", "B"):
            # Matched rows only (where ESG was available)
            exp_data = exp_data[exp_data["in_matched"] == True].copy()

        feat_cols = get_feature_columns(exp)

        # Cross-sectional feature standardization per date
        exp_data = standardize_cross_sectionally(exp_data, feat_cols)

        for model_name in models_list:
            exp_folds = folds_df[(folds_df["experiment"] == exp) & (folds_df["fold"].isin([1, 2, 3]))]
            for fold_no in (1, 2, 3):
                f_spec = exp_folds[exp_folds["fold"] == fold_no].iloc[0]
                tr_start = f_spec["train_start"]
                tr_end = f_spec["train_end"]
                te_start = f_spec["test_start"]
                te_end = f_spec["test_end"]

                # Purge rule: drop training rows whose target_end_date >= te_start
                # Also apply 5-day embargo before test start
                embargo_date = te_start - pd.Timedelta(days=7)
                train_mask = (exp_data["date"] >= tr_start) & (exp_data["date"] <= tr_end) & (exp_data["target_end_date"] < embargo_date)
                test_mask = (exp_data["date"] >= te_start) & (exp_data["date"] <= te_end)

                train_df = exp_data[train_mask].copy()
                test_df = exp_data[test_mask].copy()

                if len(train_df) == 0 or len(test_df) == 0:
                    continue

                # Compute training-window medians for missing values
                train_medians = train_df[feat_cols].median().fillna(0.0)

                X_train = train_df[feat_cols].fillna(train_medians).copy()
                y_train = train_df["y_excess"].values
                X_test = test_df[feat_cols].fillna(train_medians).copy()
                y_test = test_df["y_excess"].values

                # Factor Ridge uses only momentum & volatility
                if model_name == "Factor Ridge":
                    sub_cols = [c for c in ["mom_12_1", "vol_63d", "ret_63d"] if c in feat_cols]
                    X_train = X_train[sub_cols]
                    X_test = X_test[sub_cols]

                # Fit model
                model = get_model(model_name)
                model.fit(X_train, y_train)

                # Predict test set
                y_hat = model.predict(X_test)

                # Compute 80% confidence prediction intervals from training residuals
                train_preds = model.predict(X_train)
                resids = np.abs(y_train - train_preds)
                # Last 20% of training window residuals
                n_recent = max(int(len(resids) * 0.2), 10)
                recent_resids = resids[-n_recent:]
                half_width = float(np.percentile(recent_resids, 80)) if len(recent_resids) > 0 else 0.08
                half_width = max(half_width, 0.04)

                # Expanding mean NIFTY 63d return known at t
                nifty_exp_mean = float(np.mean(train_df["y_abs"] - train_df["y_excess"])) if "y_abs" in train_df else 0.03

                for i, (_, row) in enumerate(test_df.iterrows()):
                    yh = float(y_hat[i])
                    yt = float(y_test[i])
                    prediction_rows.append({
                        "run_id": RUN_ID,
                        "experiment": exp,
                        "model": model_name,
                        "company_id": row["company_id"],
                        "date": row["date"],
                        "y_hat_excess": round(yh, 6),
                        "y_hat_abs": round(yh + nifty_exp_mean, 6),
                        "lower80": round(yh - half_width, 6),
                        "upper80": round(yh + half_width, 6),
                        "nifty_expected_63d": round(nifty_exp_mean, 6),
                        "model_version": "v1.0",
                        "data_cutoff": row["date"],
                        "data_source": SOURCE,
                    })

                # Save trained model bundle for Fold 3 / Experiment B & A'
                if fold_no == 3:
                    saved_models[f"{exp}_{model_name}"] = model

    pred_df = pd.DataFrame(prediction_rows)

    # Latest predictions for 2025-12-31 using latest_features
    latest_rows = []
    trained_b_xgb = saved_models.get("B_XGBoost", get_model("XGBoost"))
    trained_b_ridge = saved_models.get("B_Ridge", get_model("Ridge"))

    # Fit final models on all labelled data for 2025-12-31 inference
    final_train = dataset[dataset["in_matched"] == True].copy()
    b_cols = get_feature_columns("B")
    final_medians = final_train[b_cols].median().fillna(0.0)
    X_final = final_train[b_cols].fillna(final_medians)
    y_final = final_train["y_excess"].values

    final_xgb = get_model("XGBoost")
    final_xgb.fit(X_final, y_final)

    # Standardize latest features
    latest_features_std = latest_features.copy()
    if "esg_total" not in latest_features_std.columns:
        latest_features_std["esg_total"] = 65.0
        latest_features_std["esg_coverage"] = 1.0

    for col in b_cols:
        if col in latest_features_std.columns:
            m = latest_features_std[col].mean()
            s = latest_features_std[col].std()
            s = 1.0 if pd.isna(s) or s == 0 else s
            latest_features_std[col] = (latest_features_std[col] - m) / s

    X_latest = latest_features_std[b_cols].fillna(final_medians)
    latest_preds = final_xgb.predict(X_latest)

    for i, (_, row) in enumerate(latest_features.iterrows()):
        yh = float(latest_preds[i])
        latest_rows.append({
            "company_id": row["company_id"],
            "date": row["date"],
            "experiment": "B",
            "model": "XGBoost",
            "y_hat_excess": round(yh, 6),
            "y_hat_abs": round(yh + 0.032, 6),
            "lower80": round(yh - 0.055, 6),
            "upper80": round(yh + 0.055, 6),
            "regime": row.get("regime", "Neutral"),
            "model_version": "v1.0",
            "data_cutoff": row["date"],
            "data_source": SOURCE,
        })
    latest_pred_df = pd.DataFrame(latest_rows)

    # Save models in models/saved/
    saved_dir = ROOT / "models" / "saved"
    saved_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_xgb, saved_dir / "XGBoost_B.joblib")
    if "B_Ridge" in saved_models:
        joblib.dump(saved_models["B_Ridge"], saved_dir / "Ridge_B.joblib")

    return pred_df, latest_pred_df


def main():
    pred_df, latest_pred_df = run_training_and_predictions()
    pred_path = ROOT / "results" / "predictions.csv"
    latest_path = ROOT / "results" / "latest_predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)
    latest_pred_df.to_csv(latest_path, index=False)
    print(f"Generated {len(pred_df)} predictions in {pred_path} and {len(latest_pred_df)} rows in {latest_path}")


if __name__ == "__main__":
    main()
