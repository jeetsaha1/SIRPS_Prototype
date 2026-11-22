# SIRPS: Sustainable Investment Return Prediction System (Synthetic Prototype)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit 1.41.1](https://img.shields.io/badge/streamlit-1.41.1-red.svg)](https://streamlit.io/)
[![Tests passing](https://img.shields.io/badge/tests-81%2F81%20passing-brightgreen.svg)](tests/)
[![Data Integrity](https://img.shields.io/badge/validation-32%2F32%20checks%20passed-success.svg)](tests/validate_data.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A point-in-time, synthetic ESG-aware 63-trading-day return forecasting and risk analytics prototype designed for algorithmic pipeline and architectural validation (modeled on the Indian equity market and SEBI BRSR disclosure framework).

SIRPS provides an end-to-end econometric research harness to validate whether incorporating corporate **Environmental, Social, and Governance (ESG)** performance metrics provides measurable incremental predictive power beyond price technicals and fundamental factors under controlled simulation conditions.

> **Synthetic Prototype & Academic Honesty Disclosure:**  
> This software is a **synthetic research prototype** developed for econometric architecture, data pipeline engineering, and machine learning validation. All datasets are marked with `SYNTHETIC_DEMO=1`. The data generation pipeline plants known synthetic signals (e.g., an ESG-linked drift of strength 0.04 per standard deviation per 63 days) specifically to verify that the econometric pipeline can correctly detect and attribute signals without lookahead bias. **Statistical results demonstrate algorithmic pipeline mechanics, not real-world market evidence, trading alpha, or investment advice.**

---

## Table of Contents

- [Core Capabilities](#core-capabilities)
- [System Architecture](#system-architecture)
- [The 20 Canonical Data Contracts](#the-20-canonical-data-contracts)
- [Econometric & Mathematical Methodology](#econometric--mathematical-methodology)
  - [1. Excess Return Formulation](#1-excess-return-formulation)
  - [2. Causal Macro Regime Filter](#2-causal-macro-regime-filter)
  - [3. Purged K-Fold Cross-Validation](#3-purged-k-fold-cross-validation)
  - [4. Statistical Hypothesis Testing (Clark-West & Placebo)](#4-statistical-hypothesis-testing-clark-west--placebo)
  - [5. Additive Local Attribution](#5-additive-local-attribution)
- [Dashboard Modules (10 Pages)](#dashboard-modules-10-pages)
- [Quickstart Guide](#quickstart-guide)
  - [Installation & Virtual Environment](#installation--virtual-environment)
  - [Running the Streamlit Application](#running-the-streamlit-application)
  - [Single-Page Development Runner](#single-page-development-runner)
  - [End-to-End Pipeline Regeneration](#end-to-end-pipeline-regeneration)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Mock Fallback Architecture](#mock-fallback-architecture)
- [Project Directory Structure](#project-directory-structure)

---

## Core Capabilities

- **Strict Point-In-Time (PIT) Data Hygiene:** Prevents lookahead leakage by enforcing an `available_date <= date` constraint on all BRSR annual ESG filings and quarterly corporate earnings.
- **Leakage-Free Cross-Validation:** Features a 3-fold expanding-window cross-validation regime parameterized with a **63-day forward purge window** and a **5-day embargo** to eliminate overlapping target contamination.
- **Model Diversity & Competitive Baselines:** Implements non-trivial baseline models (Zero Return, Expanding Mean, Momentum Rank) alongside machine learning forecasters (L2 Regularized Ridge Regression, Random Forest, and XGBoost).
- **Rigorous Hypothesis Verification:** Benchmarks incremental predictive gains using out-of-sample Clark-West test statistics for nested models ($p < 0.05$) paired with a 30-run block bootstrap / surrogate placebo distribution.
- **Interactive Multi-Horizon Explainability:** Delivers global permutation importance and locally additive waterfall attributions satisfying $\text{Base Value} + \sum c_i = \hat{y}_{\text{excess}}$ within $10^{-6}$ numerical tolerance.
- **Real-Time What-If Scenario Simulator:** Allows investment analysts to interactively modify individual ESG pillars (E, S, G scores) in Streamlit and observe real-time recalculations of predicted returns and confidence intervals directly from trained ML pipelines.
- **Overlapping Cohort Portfolio Simulation:** Backtests daily-staggered 63-day cohorts comparing long-short and market-hedged allocations against the NIFTY 50 Total Return Index (TRI).

---

## System Architecture

```
                                  [ Raw Universe & SEBI BRSR Filings ]
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
    [ Market & Fundamental PIT ]                                              [ Annual ESG Facts ]
   - OHLCV Daily Prices (10 firms)                                          - BRSR Disclosures (8 metrics)
   - Corporate Actions (Splits/Divs)                                        - Percentile E, S, G Scoring
   - Quarterly Financials (7 metrics)                                       - Imputation & Trend Tracking
                 │                                                                     │
                 └──────────────────────────────────┬──────────────────────────────────┘
                                                    ▼
                                    [ Point-in-Time Feature Store ]
                                  - Technical Features (RSI, Vol, SMA)
                                  - Fundamental Factors (P/E, ROE, D/E)
                                  - ESG Features (Scores & Momentum)
                                  - 63-Day Excess Target (vs NIFTY TRI)
                                                    │
                                                    ▼
                                    [ Purged Expanding CV Splits ]
                                    (63d Purge Window + 5d Embargo)
                                                    │
                    ┌───────────────────────────────┴───────────────────────────────┐
                    ▼                                                               ▼
        [ Forecasting Models ]                                          [ Statistical Rigor ]
      - Baselines (Zero, Mean, Mom)                                   - Out-of-sample Clark-West Test
      - Ridge, Random Forest, XGBoost                                 - 30-Run Surrogate Placebo Distribution
                    │                                                               │
                    └───────────────────────────────┬───────────────────────────────┘
                                                    ▼
                                    [ Explainability & Attribution ]
                                  - Additive Local Decomposition
                                  - Global Permutation Importance
                                                    │
                                                    ▼
                              [ Interactive 10-Page Streamlit Dashboard ]
```

---

## The 20 Canonical Data Contracts

All data files in `data/` and `results/` conform strictly to standardized schemas, primary key definitions, and exact target record dimensions:

| # | Dataset File | Rows | Primary Key | Description |
| :-: | :--- | :-: | :--- | :--- |
| 1 | `data/company_master.csv` | **10** | `company_id` | NIFTY 50 universe metadata (C01–C10), ticker, sector, weights |
| 2 | `data/trading_calendar.csv` | **1,526** | `date` | NSE calendar (2020–2025) identifying active trading days and quarters |
| 3 | `data/index_daily.csv` | **1,526** | `date` | Benchmark NIFTY 50 Close, NIFTY Total Return Index (TRI), India VIX |
| 4 | `data/prices_daily.csv` | **15,260** | `company_id`, `date` | 10 firms × 1,526 days OHLCV price series |
| 5 | `data/corporate_actions.csv` | **80** | `company_id`, `ex_date` | Cash dividends, bonus issuances, and stock splits |
| 6 | `data/esg_facts.csv` | **320** | `company_id`, `fiscal_year`, `indicator_id` | 10 firms × 4 fiscal years × 8 BRSR disclosures with availability dates |
| 7 | `data/esg_scores.csv` | **40** | `company_id`, `fiscal_year` | Computed Environmental, Social, Governance, and aggregate ESG scores |
| 8 | `data/fundamentals_pit.csv` | **1,890** | `company_id`, `fiscal_year`, `quarter`, `metric_id` | 27 quarters × 10 firms × 7 balance sheet/income statement metrics |
| 9 | `data/regime_daily.csv` | **1,526** | `date` | Causal market regime classification (Bull, Bear, Neutral) |
| 10 | `data/model_dataset.csv` | **2,420** | `company_id`, `date` | 5-day sampled feature grid ($t+1$ to $t+64$) with 28 engineered predictors |
| 11 | `data/latest_features.csv` | **10** | `company_id` | Most recent available feature vector per firm for real-time inference |
| 12 | `data/fold_definitions.csv` | **15** | `experiment_id`, `fold_id` | Purged cross-validation splits (Folds 1, 2, 3 across Experiments A, A', B, C, P) |
| 13 | `results/predictions.csv` | **29,190** | `experiment_id`, `fold_id`, `model_name`, `company_id`, `date` | Out-of-sample predictions across folds, models, and experiments |
| 14 | `results/latest_predictions.csv` | **10** | `company_id` | Current forward 63-day forecasts with 80% confidence interval bands |
| 15 | `results/metrics_summary.csv` | **147** | `experiment_id`, `fold_id`, `model_name`, `metric` | MSFE, RMSE, MAE, $R^2$, Rank-IC, and Directional Accuracy evaluations |
| 16 | `results/experiment_results.csv` | **12** | `hypothesis`, `model_name` | Clark-West test statistics, $t$-values, and asymptotic $p$-values |
| 17 | `results/placebo_distribution.csv` | **31** | `run_id` | Baseline empirical metric vs 30 feature-scrambled placebo runs |
| 18 | `results/feature_importance.csv` | **228** | `experiment_id`, `model_name`, `feature_name` | Out-of-sample permutation feature importance distributions |
| 19 | `results/local_contributions.csv` | **200** | `company_id`, `fold_id`, `feature_name` | Additive local attributions: $\text{Base Value} + \sum c_i = \hat{y}_{\text{excess}}$ |
| 20 | `results/portfolio_default.csv` | **1,526** | `date` | Overlapping cohort portfolio backtest equity curve and risk metrics |

*Every table contains `SYNTHETIC_DEMO = 1` as its terminal column to ensure synthetic compliance.*

---

## Econometric & Mathematical Methodology

### 1. Excess Return Formulation

The prediction target is the forward 63-trading-day **excess return** over the market benchmark (NIFTY Total Return Index):

$$y_{i,t}^{\text{abs}} = \frac{P_{i, t+63}^{\text{adj}} - P_{i, t}^{\text{adj}}}{P_{i, t}^{\text{adj}}}$$

$$y_{i,t}^{\text{excess}} = y_{i,t}^{\text{abs}} - \frac{\text{TRI}_{t+63} - \text{TRI}_t}{\text{TRI}_t}$$

Where:
- $P_{i, t}^{\text{adj}}$ is the dividend- and split-adjusted closing stock price for company $i$ at date $t$.
- $\text{TRI}_t$ is the NIFTY 50 Total Return Index (reinvesting cash distributions at date $t$).
- Predictions are generated at the close of trading day $t$, with entry executed at the close of $t+1$ and exit executed at the close of $t+64$ (a 63-trading-day holding duration).

### 2. Causal Macro Regime Filter

Regimes are estimated strictly without lookahead bias using backward-looking indicators:
1. **Trend Criterion:** 50-day Simple Moving Average (SMA) of NIFTY 50 vs 200-day SMA.
2. **Volatility Criterion:** India VIX level ($< 15$ Low, $15\text{--}22$ Moderate, $> 22$ High).
3. **Causal Filter:** A minimum **10-consecutive-day confirmation rule** is required before a regime state switch (Bull $\leftrightarrow$ Bear $\leftrightarrow$ Neutral) is recognized, preventing transient whipsaws.

### 3. Purged K-Fold Cross-Validation

To prevent data leakage caused by serial correlation in overlapping multi-day returns:
- **Purge Window (63 Trading Days):** Any training observation whose forward prediction window $(t \to t+63)$ overlaps with the evaluation test window is purged from the training set.
- **Embargo Window (5 Trading Days):** An additional 5-day buffer is inserted immediately following the test boundary to prevent autoregressive leakage from the test period into subsequent training intervals.

### 4. Statistical Hypothesis Testing (Clark-West & Placebo)

To formally test whether ESG signals add genuine predictive capacity beyond standard financial predictors:
- **Clark-West Test for Nested Models:** Accounts for the upward parameter-estimation noise bias in larger models under the null hypothesis of equal forecast accuracy:

$$\text{CW}_t = (y_{t+h} - \hat{y}_{1, t+h})^2 - \left[ (y_{t+h} - \hat{y}_{2, t+h})^2 - (\hat{y}_{1, t+h} - \hat{y}_{2, t+h})^2 \right]$$

- **30-Run Surrogate Placebo Distribution:** Autocorrelation-preserving block bootstrap that shuffles company-level ESG assignments across 30 iterations while leaving price and fundamental series unaltered. An ESG model is statistically validated only if its empirical $\Delta \text{MSFE}$ ranks beyond the 95th percentile of the placebo distribution.

### 5. Additive Local Attribution

Every point forecast is decomposed into interpretable linear attributions:

$$\hat{y}_{i,t}^{\text{excess}} = \text{Base Value} + \sum_{j=1}^{M} c_{i,t,j}$$

Where $\text{Base Value}$ is the sample expected return and $c_{i,t,j}$ is the marginal contribution of predictor $j$. For all models, this decomposition is strictly verified to sum to $\hat{y}$ within a numerical precision tolerance of $\epsilon < 10^{-6}$.

---

## Dashboard Modules (10 Pages)

The interactive dashboard is organized into 10 modular analytical pages:

1. **Executive Dashboard (`home.py`):** High-level KPI summary cards (Total Return, Annualized Sharpe, Alpha, Mean ESG), NIFTY vs. Strategy equity curve with macro regime shading, and top ESG / Return leaders.
2. **Company Explorer (`company_explorer.py`):** Interactive OHLCV candlestick price charts with volume bars, historical corporate actions ledger (dividends, splits), and Point-in-Time financial metrics table.
3. **ESG Deep Dive (`esg_analysis.py`):** E/S/G pillar radar charts, historical ESG score progression, cross-sectional peer comparison scatter plot (ESG Score vs Return), and raw BRSR indicator disclosures.
4. **Market Regime Analysis (`market_regime.py`):** Chronological macro regime timeline with 10-day confirmation filter, India VIX overlay, Markov state transition probability matrix heatmap, and regime-stratified forecasting accuracy.
5. **Return Prediction Engine (`return_prediction.py`):** Forward 63-day return predictions across the 10 firms with 80% confidence interval bands, prediction vs. actual scatter plot with 45° reference line, and an **interactive What-If ESG slider simulation** executing live model inference.
6. **Model Benchmark & Comparison (`model_comparison.py`):** Out-of-sample leaderboard across Ridge, Random Forest, XGBoost, and Baselines. Metric toggles (MSFE, RMSE, MAE, $R^2$, Rank-IC, Directional Accuracy), Clark-West test $p$-values, and placebo test histogram.
7. **ESG Value-Add Attribution (`esg_contribution.py`):** Granular comparison of Experiment A (Technicals + Fundamentals) vs Experiment B (+ESG). Pillar-level waterfall contribution chart and rolling $\Delta \text{MSFE}$ time series.
8. **Portfolio & Risk Management (`portfolio_risk.py`):** 63-day overlapping cohort backtest wealth progression, underwater drawdown chart, risk analytics table (Sharpe, Sortino, VaR 95%, Max Drawdown), and dynamic sector weighting bar chart.
9. **Model Explainability (`explainability.py`):** Global out-of-sample permutation feature importance charts and company-level additive local waterfall charts conforming to $\text{Base} + \sum c_i = \hat{y}$.
10. **Methodology & Data Dictionary (`methodology.py`):** Mathematical formulations rendered via LaTeX, interactive schema data dictionary, and one-click CSV download buttons for all 20 canonical datasets.

---

## Quickstart Guide

### Installation & Virtual Environment

Clone the repository and prepare the virtual environment:

```powershell
# 1. Clone repository
git clone https://github.com/jeetsaha1/SIRPS_Prototype.git
cd SIRPS_prototype

# 2. Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows PowerShell
# source .venv/bin/activate    # On Linux / macOS

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Running the Streamlit Application

Launch the full interactive multi-page dashboard:

```powershell
streamlit run app.py
```

Open your browser to `http://localhost:8501`. Use the sidebar radio buttons to navigate between all 10 analytical modules.

### Single-Page Development Runner

To isolate and inspect individual page modules during development or testing:

```powershell
streamlit run dev_run.py -- home
streamlit run dev_run.py -- company_explorer
streamlit run dev_run.py -- esg_analysis
streamlit run dev_run.py -- return_prediction
streamlit run dev_run.py -- model_comparison
streamlit run dev_run.py -- explainability
```

### Deploying as a Live Public Website (With Link)

To get a public HTTPS link (like a deployed MERN web app on Vercel/Netlify) that anyone can visit in their browser:

#### Option A: Streamlit Community Cloud (Recommended — Free & 1-Click)
1. Push this repository to your GitHub account (`https://github.com/jeetsaha1/SIRPS_Prototype`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**.
4. Select your repository: `SIRPS_Prototype`, branch: `integration` (or `main`), and main file path: `app.py`.
5. Click **"Deploy!"**.
6. Streamlit automatically builds and deploys your website with a permanent public link:  
   👉 `https://sirps-prototype.streamlit.app` (or your custom subdomain).

#### Option B: Render.com / Railway (Using `render.yaml` or `Procfile`)
1. Create a free account at [render.com](https://render.com).
2. Connect your GitHub repository.
3. Select **"Web Service"** — Render automatically detects `render.yaml` and sets the build command to `pip install -r requirements.txt` and start command to `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
4. Click **Deploy**. Render gives you a public URL: `https://sirps-prototype.onrender.com`.

#### Option C: Docker Container Deployment (`Dockerfile`)
Build and run the production web container locally or on any cloud (AWS ECS, Google Cloud Run, Azure App Service, DigitalOcean):
```bash
docker build -t sirps-website .
docker run -p 8501:8501 sirps-website
```
Visit `http://localhost:8501` or your cloud server's public IP/domain.

---

### End-to-End Pipeline Regeneration

To regenerate all datasets, train the forecasting models, compute metrics, and run statistical tests from scratch:

```powershell
python data_generation/run_all.py
```

This single command executes the complete data and modeling pipeline in deterministic sequence:
1. `make_company_master.py` (C01–C10 universe)
2. `gen_calendar.py` (1,526 NSE trading days)
3. `gen_index.py` (NIFTY 50, NIFTY TRI, India VIX)
4. `gen_regime.py` (Causal macro regime classification)
5. `gen_prices.py` (15,260 OHLCV rows & 80 corporate actions)
6. `gen_esg.py` (320 BRSR facts & 40 ESG scores)
7. `gen_fundamentals.py` (1,890 quarterly accounting items)
8. `build_targets.py` & `build_model_dataset.py` (2,420 model rows & 10 latest features)
9. `build_variants.py` (Experimental feature subsets)
10. `models/splits.py` (15 purged CV fold definitions)
11. `models/train_experiments.py` (Train Ridge, RF, XGBoost; generate predictions & joblib bundles)
12. `models/evaluate.py` (Compute MSFE, RMSE, MAE, R², Rank-IC, DA metrics)
13. `models/stats_tests.py` (Compute Clark-West test & 30 placebo runs)
14. `models/explain.py` (Permutation importance & additive local contributions)
15. `common/portfolio_engine.py` (1,526-day overlapping cohort backtest)

---

## Testing & Quality Assurance

SIRPS includes an automated test harness covering both econometric data integrity and user interface rendering:

### 1. Data Contract Validation (`tests/validate_data.py`)

Verifies all 20 canonical schemas, primary keys, date formats, bounds, and point-in-time constraints:

```powershell
python tests/validate_data.py
```

**Output:**
```text
PASS company_master
PASS trading_calendar
PASS index_daily
PASS prices_daily
PASS corporate_actions
PASS esg_facts
PASS esg_scores
PASS fundamentals_pit
PASS regime_daily
PASS model_dataset
PASS latest_features
PASS fold_definitions
PASS predictions
PASS metrics_summary
PASS latest_predictions
PASS experiment_results
PASS placebo_distribution
PASS feature_importance
PASS local_contributions
PASS portfolio_default
PASS company_master_count
PASS company_master_keys
PASS calendar_keys
PASS ohlc_rules
PASS esg_ranges
PASS esg_missing_codes
PASS target_recomputation
PASS target_dates
PASS target_values
PASS prediction_keys
PASS prediction_range
PASS point_in_time_esg
All 32 data checks passed.
```

### 2. Pytest Automated Test Suite (`tests/`)

Runs unit tests, econometric invariant checks, and headless Streamlit `AppTest` smoke tests across all 10 pages in both mock and real synthetic modes:

```powershell
pytest -q tests/
```

**Output:**
```text
81 passed in ~55s (100% pass rate)
- 20 Streamlit AppTest Headless UI Smoke Tests (10 pages x Mock & Real modes): PASS
- 5 Market Data & NSE Calendar Tests: PASS
- 3 ESG Indicators & Scoring Formula Tests: PASS
- 3 Causal Regime Classification & Truncation Causality Tests: PASS
- 2 Cohort Portfolio Simulation & Benchmark Alignment Tests: PASS
- 3 Dataset Schema, 5-Day Sampling & 1e-9 Target Recomputation Tests: PASS
- 4 Point-In-Time Leakage & 5-Day Embargo Tests: PASS
- 4 ML Split Purging, Baselines & 1e-6 Local Additivity Tests: PASS
- 1 Metric Recomputation Tolerance Test: PASS
- 3 P1 Integration & G0-G5 Gate Checks: PASS
- 32 Schema, Interface & Master Pipeline Tests: PASS
```

---

## Release Preparation & Fresh-Clone Verification (`v0.1-demo`)

The prototype is prepared for clean demonstration from a fresh clone offline:

1. **Gate G5 Release Checklist:**
   - [x] All 20 canonical datasets generated and committed with `SYNTHETIC_DEMO=1`.
   - [x] All 10 Streamlit pages load without exceptions in both Mock (`$env:SIRPS_USE_MOCK="1"`) and Real Synthetic modes.
   - [x] `python tests/validate_data.py` passes 32/32 domain checks.
   - [x] `pytest tests/` passes 81/81 test cases in under 60 seconds.
   - [x] Clear synthetic prototype disclaimers displayed consistently across all pages.
2. **Tagging Release:**
   ```powershell
   git tag -a v0.1-demo -m "SIRPS Prototype v0.1-demo: feature-complete 10-page synthetic prototype"
   ```

---

## Mock Fallback Architecture

To ensure high availability and resilient development without broken pages:
- The data access layer (`common/io.py`) provides an intelligent fallback mechanism: if any primary CSV file in `data/` or `results/` is temporarily missing, it transparently loads a synthetic seed from `data/mock/`.
- In mock mode, a prominent warning banner is rendered at the top of the interface:
  > **Demo Mode Notice:** Displaying synthetic preview data.
- To explicitly force mock mode for testing:
  ```powershell
  $env:SIRPS_USE_MOCK="1"
  streamlit run app.py
  ```
- In default production operation with real pipeline data present, `common.io.is_using_mock` evaluates to `False` across all 20 tables.

---

## Project Directory Structure

```text
SIRPS_prototype/
├── app.py                      # Main 10-page Streamlit application entry point
├── dev_run.py                  # Single-page developer runner
├── requirements.txt            # Pinned Python package dependencies
├── README.md                   # Project documentation and specifications
│
├── common/                     # Shared architectural contracts & utilities
│   ├── contracts.py            # Primary keys, seeds, schemas, and table constants
│   ├── io.py                   # PIT-safe file reader with automatic mock fallback
│   ├── metrics.py              # MSFE, RMSE, MAE, R², Rank-IC, Directional Accuracy
│   ├── portfolio_engine.py     # Overlapping 63-day cohort backtest engine
│   ├── regime.py               # Causal macro regime filter (10-day confirmation)
│   ├── scoring.py              # BRSR percentile rank ESG scoring engine
│   └── ui.py                   # Shared theme, CSS styling, and warning banners
│
├── data/                       # Production datasets (20 canonical tables)
│   ├── company_master.csv      # 10 NIFTY 50 universe firms (C01–C10)
│   ├── trading_calendar.csv    # 1,526 NSE trading days (2020–2025)
│   ├── index_daily.csv         # NIFTY 50 close, TRI, and India VIX
│   ├── prices_daily.csv        # 15,260 OHLCV price series
│   ├── corporate_actions.csv   # 80 corporate split and dividend events
│   ├── esg_facts.csv           # 320 raw BRSR indicator disclosures
│   ├── esg_scores.csv          # 40 computed E/S/G and aggregate scores
│   ├── fundamentals_pit.csv    # 1,890 point-in-time quarterly financial items
│   ├── regime_daily.csv        # 1,526 causal market regime records
│   ├── model_dataset.csv       # 2,420 engineered 5-day step model rows
│   ├── latest_features.csv     # 10 current inference feature vectors
│   ├── fold_definitions.csv    # 15 purged cross-validation fold boundaries
│   └── mock/                   # Complete fallback seed datasets
│
├── data_generation/            # Pipeline data synthesis and feature engineering
│   ├── run_all.py              # Master pipeline orchestrator script
│   ├── make_company_master.py  # Generates company universe
│   ├── gen_calendar.py         # Builds NSE trading calendar
│   ├── gen_index.py            # Generates index and TRI time series
│   ├── gen_prices.py           # Generates geometric Brownian motion prices & actions
│   ├── gen_regime.py           # Builds causal regime series
│   ├── gen_esg.py              # Synthesizes BRSR facts and scores
│   ├── gen_fundamentals.py     # Generates PIT quarterly financial records
│   ├── feat_technical.py       # Technical indicators (RSI, Vol, SMA)
│   ├── feat_market.py          # Market beta and relative strength features
│   ├── feat_esg.py             # ESG level, momentum, and pillar features
│   ├── feat_fundamental.py     # Financial ratios (P/E, ROE, D/E)
│   ├── build_targets.py        # 63-day forward excess return calculations
│   ├── build_model_dataset.py  # Merges features, targets, and dates
│   ├── build_variants.py       # Constructs experiment subset flag mappings
│   └── make_mock_data.py       # Generates baseline mock fallback files
│
├── models/                     # Machine learning training & econometric evaluation
│   ├── baselines.py            # Zero, Expanding Mean, and Momentum benchmarks
│   ├── splits.py               # 3-fold purged cross-validation splits
│   ├── train_experiments.py    # Trains Ridge, RF, XGBoost; saves models
│   ├── evaluate.py             # Computes metrics across folds and models
│   ├── stats_tests.py          # Clark-West tests & 30-run placebo distribution
│   ├── explain.py              # Permutation importance & local attributions
│   ├── predict_api.py          # Real-time inference API for What-If simulator
│   └── saved/                  # Serialized .joblib trained model artifacts
│
├── modules/                    # Streamlit dashboard pages (10 UI modules)
│   ├── home.py                 # Executive Dashboard & KPI summary
│   ├── company_explorer.py     # Interactive candlestick & financial explorer
│   ├── esg_analysis.py         # ESG pillar radars, trends, and disclosures
│   ├── market_regime.py        # Regime timeline, transition matrix, and VIX
│   ├── return_prediction.py    # Predictions, 80% CI bands, and What-If slider
│   ├── model_comparison.py     # Leaderboard, metric toggles, and placebo test
│   ├── esg_contribution.py     # Experiment A vs B delta & waterfall attribution
│   ├── portfolio_risk.py       # Cohort backtest equity curve and risk metrics
│   ├── explainability.py       # Global importance & local additive waterfall
│   └── methodology.py          # Mathematical formulas & data dictionary
│
├── results/                    # Model evaluation, prediction, and test outputs
│   ├── predictions.csv         # 29,190 out-of-sample predictions
│   ├── latest_predictions.csv  # 10 current forecasts with 80% CI bounds
│   ├── metrics_summary.csv     # 147 performance metric records
│   ├── experiment_results.csv  # Clark-West statistical test results
│   ├── placebo_distribution.csv# 31 empirical and placebo test runs
│   ├── feature_importance.csv  # 228 permutation importance scores
│   ├── local_contributions.csv # 200 additive local attribution entries
│   └── portfolio_default.csv   # 1,526-day cohort backtest equity series
│
└── tests/                      # Automated test harness
    ├── validate_data.py        # 31 domain data integrity checks
    ├── test_data_pipeline.py   # Pipeline execution and schema tests
    ├── test_models.py          # Additivity, split, and baseline tests
    └── test_pages_smoke.py     # Headless Streamlit AppTest smoke tests (10 pages)
```

---

## License

This project is licensed under the [MIT License](LICENSE).
