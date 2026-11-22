import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from common.io import load
from common.ui import demo_banner, init_state, company_selector
from models.predict_api import predict_single


def render() -> None:
    init_state()
    demo_banner(["feature_importance", "local_contributions", "company_master", "latest_features"])
    st.title("Model Explainability & Factor Attribution")

    feat_imp = load("feature_importance")
    local_contrib = load("local_contributions")
    company_df = load("company_master")
    latest_feat = load("latest_features")

    c1, c2, c3 = st.columns(3)
    exp_choice = c1.selectbox("Experiment", sorted(feat_imp["experiment"].unique()), index=2 if "B" in feat_imp["experiment"].unique() else 0)
    model_choice = c2.selectbox("Model", sorted(feat_imp["model"].unique()))
    cid = c3.selectbox("Company for Local Explanation", sorted(company_df["company_id"].unique()))

    tab1, tab2, tab3 = st.tabs(["Global Feature Importance", "Local Additive Attribution (Waterfall)", "Interactive What-If Simulator"])

    with tab1:
        st.subheader(f"Global Permutation Feature Importance ({model_choice}, Experiment {exp_choice})")
        sub_imp = feat_imp[(feat_imp["experiment"] == exp_choice) & (feat_imp["model"] == model_choice)].sort_values("importance", ascending=True)

        fig_imp = px.bar(
            sub_imp,
            x="importance",
            y="feature",
            color="feature_group",
            orientation="h",
            title=f"Global Factor Importance Ranked by Drop in Out-of-Sample Performance",
            labels={"importance": "Permutation Importance (Loss Degradation)", "feature": "Feature"}
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # Group-level aggregation
        grp_agg = sub_imp.groupby("feature_group")["importance"].sum().reset_index()
        fig_pie = px.pie(grp_agg, values="importance", names="feature_group", title="Total Factor Group Importance Share")
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.subheader(f"Local Factor Contribution for {cid}")
        sub_local = local_contrib[(local_contrib["company_id"] == cid) & (local_contrib["model"] == "Ridge")].sort_values("contribution", key=abs, ascending=False)
        
        if len(sub_local) > 0:
            base_val = sub_local["base_value"].iloc[0]
            y_hat = sub_local["y_hat_excess"].iloc[0]

            # Top 3 drivers sentence
            top_3_feats = sub_local.head(3)["feature"].tolist()
            st.success(f"📌 **Key Finding**: The primary return drivers for **{cid}** are **{top_3_feats[0]}**, **{top_3_feats[1]}**, and **{top_3_feats[2]}**, combining with base value `{base_val:+.4f}` to yield predicted excess return `{y_hat:+.2%}`.")

            # Waterfall chart
            waterfall_feats = ["Base Value"] + sub_local.head(8)["feature"].tolist() + ["Forecast"]
            waterfall_vals = [base_val] + sub_local.head(8)["contribution"].tolist() + [y_hat]
            measures = ["absolute"] + ["relative"] * 8 + ["total"]

            fig_wf = go.Figure(go.Waterfall(
                name="Attribution",
                orientation="v",
                measure=measures,
                x=waterfall_feats,
                textposition="outside",
                text=[f"{v:+.3f}" for v in waterfall_vals],
                y=waterfall_vals,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            fig_wf.update_layout(title=f"Factor Attribution Waterfall for {cid} (Exact Additive Identity)", height=450, template="plotly_white")
            st.plotly_chart(fig_wf, use_container_width=True)

            st.dataframe(sub_local[["feature", "feature_group", "contribution", "base_value", "y_hat_excess"]], use_container_width=True)
        else:
            st.info("No local contributions found.")

    with tab3:
        st.subheader(f"Interactive What-If Scenario Simulator ({cid})")
        st.markdown("Adjust key firm factors below to observe simulated changes in predicted quarterly excess return in real-time.")

        c_w1, c_w2, c_w3 = st.columns(3)
        w_mom = c_w1.slider("12-1 Month Momentum (mom_12_1)", -0.40, 0.60, 0.05, step=0.02)
        w_ret63 = c_w2.slider("Trailing 63-Day Return (ret_63d)", -0.30, 0.40, 0.02, step=0.01)
        w_vol63 = c_w3.slider("63-Day Volatility (vol_63d)", 0.10, 0.60, 0.25, step=0.02)

        c_w4, c_w5 = st.columns(2)
        w_esg = c_w4.slider("ESG Total Percentile Score (esg_total)", 20.0, 95.0, 65.0, step=1.0)
        w_cov = c_w5.slider("BRSR Reporting Coverage Ratio", 0.50, 1.00, 0.88, step=0.02)

        test_feat_dict = {
            "mom_12_1": w_mom,
            "ret_63d": w_ret63,
            "vol_63d": w_vol63,
            "esg_total": w_esg,
            "esg_coverage": w_cov,
            "ret_5d": 0.01,
            "ret_21d": 0.02,
            "ret_126d": 0.05,
            "price_ma50_ratio": 1.02,
            "price_ma200_ratio": 1.05,
            "beta_126d": 1.0,
            "rsi_14": 52.0,
            "volume_ratio": 1.0,
            "maxdd_126d": -0.08,
            "nifty_ret_21d": 0.01,
            "nifty_ret_63d": 0.03,
            "nifty_vol_21d": 0.14,
            "india_vix": 14.5,
            "vix_chg_21d": 0.0,
            "nifty_drawdown": -0.03,
        }

        re_pred = predict_single(test_feat_dict, experiment=exp_choice, model_name=model_choice)
        st.metric("Simulated Predicted 63-Day Excess Return", f"{re_pred:+.2%}", delta=f"{(re_pred - 0.01):+.2%} vs Baseline")
