import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

SOURCE = "SYNTHETIC_DEMO"


def simulate_portfolio(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    index: pd.DataFrame,
    regime: pd.DataFrame = None,
    company_master: pd.DataFrame = None,
    k: int = 3,
    cost: float = 0.003,
    rf: float = 0.065,
    n_random: int = 200,
    seed: int = 42,
) -> dict:
    """Staggered cohort portfolio simulator conforming to Blueprint Phase 4:
    - New cohort formed every 21 trading days from out-of-sample forecasts.
    - Top-k by forecast excess return, equal weight (1/k).
    - Entry at close t+1, exit at close t+64 (held exactly 63 trading days).
    - Amortized transaction costs 2 * cost over the 63 holding days (zero cost == gross return).
    - Portfolio daily return = simple average of returns across all currently active cohorts.
    - Benchmarks: NIFTY TRI, Equal-Weight Universe, and 200 random top-k draws (5th-95th percentile).
    - Returns dict: daily, metrics, holdings, by_regime.
    """
    rng = np.random.default_rng(seed)

    # 1. Align Dates & Universe
    idx_df = index.sort_values("date").reset_index(drop=True)
    dates = idx_df["date"].values
    n_days = len(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # NIFTY TRI daily returns
    nifty_tri = idx_df["nifty_tri"].values
    nifty_ret = np.zeros(n_days)
    nifty_ret[1:] = (nifty_tri[1:] / nifty_tri[:-1]) - 1.0

    # Stock price pivot & returns: shape (n_days, n_companies)
    px_pivot = prices.pivot(index="date", columns="company_id", values="adj_close").reindex(idx_df["date"]).ffill().bfill()
    companies = list(px_pivot.columns)
    n_comp = len(companies)
    cid_to_col = {cid: c for c, cid in enumerate(companies)}
    stock_px_matrix = px_pivot.values
    stock_returns = np.zeros_like(stock_px_matrix)
    stock_returns[1:] = (stock_px_matrix[1:] / stock_px_matrix[:-1]) - 1.0

    # Equal-weight universe daily return
    equal_weight_ret = np.nanmean(stock_returns, axis=1)

    # 2. Prediction Lookups for Cohort Formation
    # Sort predictions by date
    preds_clean = predictions.copy()
    if "date" in preds_clean.columns:
        preds_clean["date"] = pd.to_datetime(preds_clean["date"])

    # Determine available prediction dates
    pred_dates = sorted(preds_clean["date"].unique())
    pred_date_set = set(pred_dates)

    # Form new cohort every 21 trading days starting after warm-up
    # We want cohorts formed from the out-of-sample prediction period
    cal_indices = np.arange(n_days)
    # Find start date where predictions exist
    first_pred_idx = date_to_idx.get(pred_dates[0], 252) if pred_dates else 252
    cohort_formation_indices = [idx for idx in range(first_pred_idx, n_days - 64, 21)]

    daily_cost = (2.0 * cost) / 63.0

    cohorts = []
    # For each cohort formation date t
    for t_idx in cohort_formation_indices:
        t_date = dates[t_idx]
        # Get predictions for date t_date (or latest available up to t_date)
        if t_date in pred_date_set:
            p_slice = preds_clean[preds_clean["date"] == t_date]
        else:
            # Most recent prediction date <= t_date
            prior_dates = [pd for pd in pred_dates if pd <= t_date]
            p_slice = preds_clean[preds_clean["date"] == prior_dates[-1]] if prior_dates else preds_clean.head(10)

        # Rank by predicted excess return
        pred_col = "y_hat" if "y_hat" in p_slice.columns else "predicted_excess_return"
        if pred_col not in p_slice.columns:
            pred_col = p_slice.select_dtypes(include=[np.number]).columns[0]

        top_firms = p_slice.sort_values(by=pred_col, ascending=False).drop_duplicates(subset=["company_id"]).head(k)
        selected_cids = top_firms["company_id"].tolist()
        selected_cols = [cid_to_col[cid] for cid in selected_cids if cid in cid_to_col]

        entry_idx = t_idx + 1
        exit_idx = t_idx + 64

        cohorts.append({
            "formation_date": t_date,
            "entry_date": dates[entry_idx],
            "exit_date": dates[exit_idx],
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "companies": selected_cids,
            "cols": selected_cols,
            "top_df": top_firms,
        })

    # 3. Compute Portfolio Daily Returns from Active Cohorts
    # Each cohort m is active on trading days tau where entry_idx < tau <= exit_idx (63 days)
    strategy_ret = equal_weight_ret.copy()
    active_cohort_counts = np.zeros(n_days, dtype=int)
    cohort_net_returns_sum = np.zeros(n_days)

    for c in cohorts:
        cols = c["cols"]
        if not cols:
            continue
        # Days tau in (entry_idx, exit_idx]
        tau_start = c["entry_idx"] + 1
        tau_end = c["exit_idx"] + 1
        # Daily return of this cohort
        c_daily_gross = np.nanmean(stock_returns[tau_start:tau_end, cols], axis=1)
        c_daily_net = c_daily_gross - daily_cost

        cohort_net_returns_sum[tau_start:tau_end] += c_daily_net
        active_cohort_counts[tau_start:tau_end] += 1

    # Where active cohorts exist, strategy return is the simple average
    has_active = active_cohort_counts > 0
    strategy_ret[has_active] = cohort_net_returns_sum[has_active] / active_cohort_counts[has_active]

    # 4. 200 Random Top-k Benchmark Draws
    random_daily_matrix = np.zeros((n_random, n_days))
    all_col_indices = np.arange(n_comp)

    for r in range(n_random):
        r_ret_sum = np.zeros(n_days)
        r_counts = np.zeros(n_days, dtype=int)
        for c in cohorts:
            # Random k companies
            rand_cols = rng.choice(all_col_indices, size=min(k, n_comp), replace=False)
            tau_start = c["entry_idx"] + 1
            tau_end = c["exit_idx"] + 1
            c_rand_net = np.nanmean(stock_returns[tau_start:tau_end, rand_cols], axis=1) - daily_cost
            r_ret_sum[tau_start:tau_end] += c_rand_net
            r_counts[tau_start:tau_end] += 1

        r_daily = equal_weight_ret.copy()
        r_has = r_counts > 0
        r_daily[r_has] = r_ret_sum[r_has] / r_counts[r_has]
        random_daily_matrix[r] = r_daily

    # 5. Cumulative Wealth Curves (Rebased to Rs 100)
    strat_cum = 100.0 * np.cumprod(1.0 + strategy_ret)
    nifty_cum = 100.0 * np.cumprod(1.0 + nifty_ret)
    equal_cum = 100.0 * np.cumprod(1.0 + equal_weight_ret)
    rand_cum = 100.0 * np.cumprod(1.0 + random_daily_matrix, axis=1)

    daily_df = pd.DataFrame({
        "date": idx_df["date"],
        "strategy": strat_cum,
        "nifty_tri": nifty_cum,
        "equal_weight": equal_cum,
        "random_mean": np.mean(rand_cum, axis=0),
        "random_p05": np.percentile(rand_cum, 5, axis=0),
        "random_p95": np.percentile(rand_cum, 95, axis=0),
        "data_source": SOURCE,
    })

    # 6. Summary Metrics
    # Evaluate across the active evaluation period
    eval_mask = has_active if has_active.sum() > 252 else np.ones(n_days, dtype=bool)
    n_eval_days = int(eval_mask.sum())
    strat_eval_ret = strategy_ret[eval_mask]

    ann_strat_ret = float(((strat_cum[eval_mask][-1] / strat_cum[eval_mask][0]) ** (252.0 / max(n_eval_days, 1))) - 1.0)
    ann_strat_vol = float(np.std(strat_eval_ret) * np.sqrt(252))
    sharpe = (ann_strat_ret - rf) / ann_strat_vol if ann_strat_vol > 0 else 0.0

    running_peak = np.maximum.accumulate(strat_cum)
    dd = (strat_cum - running_peak) / running_peak
    max_dd = float(np.min(dd))

    # Proper Turnover & Costs Paid
    # In each cohort entry/exit, 100% of the cohort capital turns over.
    n_cohorts_total = len(cohorts)
    # Annualized turnover rate = (n_cohorts / years) * (1 / cohorts_in_steady_state)
    years = n_days / 252.0
    turnover = round(float((n_cohorts_total / years) * (1.0 / 3.0)), 2)
    # Costs paid for initial 100,000 capital
    costs_paid = round(float(2.0 * cost * (n_cohorts_total / 3.0) * 100000.0), 2)

    metrics = {
        "annualized_return": round(ann_strat_ret, 4),
        "annualized_volatility": round(ann_strat_vol, 4),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4),
        "turnover": turnover,
        "costs_paid": costs_paid,
        "cohort_count": n_cohorts_total,
    }

    # 7. Latest Cohort Holdings
    if cohorts:
        latest_c = cohorts[-1]
        top_df = latest_c["top_df"].copy()
        pred_col = "y_hat" if "y_hat" in top_df.columns else "predicted_excess_return"
        if pred_col not in top_df.columns:
            pred_col = top_df.select_dtypes(include=[np.number]).columns[0]

        holdings_rows = []
        n_firms = len(top_df)
        equal_w = 1.0 / n_firms if n_firms > 0 else 0.0
        for _, r in top_df.iterrows():
            cid = r["company_id"]
            holdings_rows.append({
                "company_id": cid,
                "weight": equal_w,
                "predicted_excess_return": round(float(r[pred_col]), 4),
                "entry_date": pd.Timestamp(latest_c["entry_date"]).strftime("%Y-%m-%d"),
                "exit_date": pd.Timestamp(latest_c["exit_date"]).strftime("%Y-%m-%d"),
                "data_source": SOURCE,
            })
        holdings = pd.DataFrame(holdings_rows)
        if company_master is not None and "symbol" in company_master.columns:
            sym_map = dict(zip(company_master["company_id"], company_master["symbol"]))
            sec_map = dict(zip(company_master["company_id"], company_master["sector"]))
            holdings["symbol"] = holdings["company_id"].map(sym_map)
            holdings["sector"] = holdings["company_id"].map(sec_map)
    else:
        holdings = pd.DataFrame(columns=["company_id", "weight", "predicted_excess_return", "entry_date", "exit_date", "data_source"])

    # 8. By-Regime Breakdown
    regime_df = regime
    if regime_df is None:
        reg_file = ROOT / "data" / "regime_daily.csv"
        if reg_file.exists():
            regime_df = pd.read_csv(reg_file, parse_dates=["date"])

    if regime_df is not None and "confirmed_regime" in regime_df.columns:
        m_reg = pd.merge(pd.DataFrame({"date": idx_df["date"], "strat_ret": strategy_ret}), regime_df, on="date", how="left")
        by_reg_rows = []
        for reg_name in ["Bull", "Neutral", "Bear"]:
            sub = m_reg[m_reg["confirmed_regime"] == reg_name]
            n_reg_dates = len(sub)
            if n_reg_dates > 5:
                sub_ret = sub["strat_ret"].values
                ann_r = float(np.mean(sub_ret) * 252.0)
                ann_v = float(np.std(sub_ret) * np.sqrt(252.0))
                sh = float((ann_r - rf) / ann_v) if ann_v > 0 else 0.0
                # Inferential badge rule: >= 60 dates and >= 2 episodes
                badge = "Inferential" if n_reg_dates >= 60 else "Descriptive only"
                by_reg_rows.append({
                    "regime": reg_name,
                    "trading_days": n_reg_dates,
                    "annualized_return": round(ann_r, 4),
                    "annualized_volatility": round(ann_v, 4),
                    "sharpe_ratio": round(sh, 2),
                    "evidence_level": badge,
                    "data_source": SOURCE,
                })
        by_regime = pd.DataFrame(by_reg_rows)
    else:
        by_regime = pd.DataFrame(columns=["regime", "trading_days", "annualized_return", "annualized_volatility", "sharpe_ratio", "evidence_level", "data_source"])

    return {
        "daily": daily_df,
        "metrics": metrics,
        "holdings": holdings,
        "by_regime": by_regime,
    }


def main():
    preds = pd.read_csv(ROOT / "results" / "predictions.csv", parse_dates=["date"])
    prices = pd.read_csv(ROOT / "data" / "prices_daily.csv", parse_dates=["date"])
    idx = pd.read_csv(ROOT / "data" / "index_daily.csv", parse_dates=["date"])
    regime = pd.read_csv(ROOT / "data" / "regime_daily.csv", parse_dates=["date"])
    comp = pd.read_csv(ROOT / "data" / "company_master.csv")

    res = simulate_portfolio(preds, prices, idx, regime=regime, company_master=comp)
    df = res["daily"]
    output = ROOT / "results" / "portfolio_default.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} portfolio simulation rows in {output}")
    print(f"Metrics: {res['metrics']}")


if __name__ == "__main__":
    main()
