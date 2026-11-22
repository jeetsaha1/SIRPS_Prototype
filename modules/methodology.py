import streamlit as st
import pandas as pd

from common.io import load
from common.schemas import FILES
from common.ui import demo_banner, init_state


def render() -> None:
    init_state()
    demo_banner(["company_master"])
    st.title("Research Methodology & Governance Protocol")
    st.caption("Pre-Implementation Baseline & Pre-Registered Research Protocol (Version 2.0)")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Econometric Target & Protocol",
        "2. Implementation Status Matrix",
        "3. Live Schema Data Dictionary",
        "4. Synthetic Generator Specs",
        "5. Academic Limitations & Ethics",
    ])

    with tab1:
        st.subheader("Primary Target & Timing Formulation")
        st.markdown("""
- **Primary Target**: 63-trading-day forward log excess return versus the NIFTY 50 Total Return Index (TRI):
  $$y_{\\text{excess}}(i, t) = \\ln\\left(\\frac{P_{i, t+64}}{P_{i, t+1}}\\right) - \\ln\\left(\\frac{\\text{TRI}_{t+64}}{\\text{TRI}_{t+1}}\\right)$$
- **Execution Timing**: Decision made after close of day $t$. Realistic entry at close of $t+1$; exit at close of $t+64$ (63 trading days held). Day $t$ close is non-tradable using information through $t$.
- **Sampling Frequency**: One observation per company every 5 trading days after 252-day feature warm-up.
- **Nested Model Comparisons**:
  - **Experiment A**: Baseline non-ESG features, full history.
  - **Experiment A'**: Non-ESG features on matched ESG-available rows.
  - **Experiment B**: Model with ESG features on identical matched rows.
  - **Experiment C**: Firm-identity control (sector & expanding firm mean).
  - **Experiment P**: Shuffled-ESG placebo test (30 permutations).
        """)

    with tab2:
        st.subheader("Implemented vs. Simulated vs. Future System Matrix")
        st.table({
            "Layer": [
                "Architecture & Storage",
                "Market & Economic Data",
                "ESG Reporting Data",
                "Machine Learning Models",
                "Statistical Inference",
                "Portfolio Backtesting",
                "Production Serving",
            ],
            "Status": [
                "Implemented",
                "Simulated / Calibrated",
                "Simulated / Calibrated",
                "Implemented",
                "Implemented",
                "Implemented",
                "Future Work",
            ],
            "Details": [
                "20 schema contracts, point-in-time joins, purged walk-forward folds, Streamlit dashboard",
                "1,526 trading days, 10-firm daily prices, causal 10-day confirmed regime filter",
                "8 pre-registered BRSR indicators, percentile rank scoring, non-zero-filled missing codes",
                "Baselines (Zero, ExpMean, MomRank), Ridge, Random Forest, XGBoost with frozen hyperparameters",
                "Clark-West test for nested models, HAC standard errors, block-bootstrap intervals, placebo runs",
                "Staggered 63-day cohorts rebalanced every 21 days with 0.3% one-way transaction costs",
                "Live NSE exchange feeds, production XBRL filing parsers, order execution management",
            ],
        })

    with tab3:
        st.subheader("Central Data Schema Contract (20 Schemas)")
        st.markdown("All files adhere strictly to the shared contract and end with mandatory `data_source = SYNTHETIC_DEMO`.")
        for name, spec in FILES.items():
            with st.expander(f"📁 {spec['folder']}/{spec['file']}  ({name})"):
                st.write(f"**Target Location**: `{spec['folder']}/{spec['file']}`")
                st.write(f"**Date Columns Parsed**: `{spec.get('dates', [])}`")
                st.code(", ".join(spec["columns"]))

    with tab4:
        st.subheader("Synthetic Calibration & Planted Alpha Parameters")
        st.markdown("""
- **Master Seed**: `SEED = 42` (ensures exact deterministic reproducibility).
- **Planted Momentum Drift**: `0.02` per standard deviation per 63 trading days.
- **Planted Quality Drift**: `0.02` per standard deviation per 63 trading days.
- **Planted ESG Drift**: `0.04` per standard deviation per 63 trading days (injected strictly after filing `available_date`).
- **Calendar Period**: 2020-01-01 to 2025-12-31 (~1,526 trading days).
        """)

    with tab5:
        st.warning("⚠️ **Academic Research Prototype Disclaimer**")
        st.markdown("""
1. **No Investment Advice**: This system is designed solely for academic and pipeline research. Nothing here constitutes investment advice, stock recommendations, or financial promotion.
2. **Synthetic Provenance**: All price histories, BRSR disclosures, financial accounting items, and model results are synthetically generated.
3. **Honest Reporting of Null Results**: If an econometric model fails to beat the zero-return baseline, or if ESG factors add no statistically significant alpha, the finding is reported without seed-hunting.
        """)
