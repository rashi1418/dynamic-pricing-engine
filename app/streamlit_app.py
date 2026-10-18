"""Pricing dashboard.

Run from the project root, after `python run_pipeline.py`:
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, feature_engineering as fe, forecasting as fc, optimization as opt  # noqa: E402

st.set_page_config(page_title="Retail Price Optimization", layout="wide")


@st.cache_data
def load_tables():
    panel = pd.read_parquet(config.PANEL_FILE)
    fut = fe.apply_category_levels(pd.read_parquet(config.DATA_PROCESSED / "next_week_features.parquet"))
    sel = pd.read_csv(config.RESULTS_DIR / "decision_elasticity.csv")
    segd = pd.read_csv(config.RESULTS_DIR / "segment_elasticity.csv")
    metrics = json.loads((config.RESULTS_DIR / "metrics.json").read_text())
    skipped = pd.read_csv(config.RESULTS_DIR / "not_recommended_products.csv")
    return panel, fut, sel, segd, metrics, skipped


@st.cache_resource
def load_model():
    m = xgb.XGBRegressor()
    m.load_model(config.MODELS_DIR / "xgb_production.json")
    return m


panel, fut, sel, segd, metrics, skipped = load_tables()
model = load_model()
calib = metrics["production"]["bias_correction"]
next_week = metrics["production"]["next_week"]

st.title("Dynamic Pricing Engine for Retail Demand Optimization")
st.caption("dunnhumby 'Breakfast at the Frat'. Forecast = XGBoost; price response = promotion-identified log-log "
           "elasticity (+ shrinkage, store segment); decision = bounded SciPy optimization with a gradual weekly price path. "
           "Profit uses an ASSUMED unit cost.")

# ---------------------------------------------------------------- sidebar inputs
# only products that are still on the shelf (see reports/results/not_recommended_products.csv)
products = sel[~sel["upc"].isin(skipped["upc"]) & sel["upc"].isin(fut["upc"])].sort_values(["category", "description"])
labels = {r.upc: f"{r.description} ({r.category.title()}, UPC {r.upc})" for r in products.itertuples()}
upc = st.sidebar.selectbox("Product", list(labels), format_func=labels.get)
with st.sidebar.expander(f"{len(skipped)} products not priced"):
    st.dataframe(skipped[["product", "reason"]], hide_index=True)
stores = ["All stores (chain)"] + sorted(fut.loc[fut["upc"] == upc, "store_num"].astype(int).unique().tolist())
store = st.sidebar.selectbox("Store", stores)

rows = fut[fut["upc"] == upc].copy()
if store != stores[0]:
    rows = rows[rows["store_num"].astype(int) == store]
w = rows["baseline_units"].clip(lower=1e-6)
p_ref = float(np.average(rows["base_price"], weights=w))

st.sidebar.markdown("---")
current_price = st.sidebar.number_input("Current regular price ($)", min_value=0.1,
                                        value=round(p_ref, 2), step=0.05)
promo = st.sidebar.selectbox("Promotion state next week", ["No promotion", "Feature ad", "In-store display",
                                                           "Feature + display"])
margin = st.sidebar.slider("ASSUMED gross margin at current price", 0.05, 0.60, config.ASSUMED_GROSS_MARGIN, 0.05)
max_change = st.sidebar.slider("Max price change (+/-)", 0.05, 0.30, config.MAX_PRICE_CHANGE, 0.05)
vol = st.sidebar.slider("Max expected volume loss", 0.02, 0.50, config.MAX_VOLUME_LOSS, 0.01)
objective = st.sidebar.radio("Objective", ["profit", "revenue"])

# ---------------------------------------------------------------- forecast at the chosen price
feat_flag = int(promo in ("Feature ad", "Feature + display"))
disp_flag = int(promo in ("In-store display", "Feature + display"))
scale = current_price / p_ref  # apply the same % change to every store's own price
scen = fe.set_price_scenario(rows, price=rows["base_price"] * scale, base_price=rows["base_price"] * scale,
                             feature=feat_flag, display=disp_flag)
Q0 = float((fc.predict_units(model, scen) * calib).sum())

e_row = sel[sel["upc"] == upc].iloc[0]
E, E_se, source = e_row["elasticity"], e_row["se"], e_row["source"]
if store != stores[0] and not np.isnan(E):
    segment = str(rows["store_segment"].iloc[0])
    sr = segd[(segd["upc"] == upc) & (segd["segment"] == segment)]
    if len(sr):
        E = float(sr["segment_elasticity"].iloc[0])
        source = f"{source}; adjusted for {segment.lower()} stores"

# ---------------------------------------------------------------- history
hist = panel[panel["upc"] == upc]
if store != stores[0]:
    hist = hist[hist["store_num"] == store]
h = hist.groupby("week").agg(units=("units", "sum"), price=("price", "mean"), base=("base_price", "mean"),
                             promo=("any_promo", "mean"))
c1, c2 = st.columns(2)
with c1:
    st.subheader("1. Historical demand")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(h.index, h["units"], lw=1)
    ax.set_ylabel("units / week")
    st.pyplot(fig)
with c2:
    st.subheader("2. Historical price")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(h.index, h["price"], lw=1, label="avg shelf price")
    ax.plot(h.index, h["base"], lw=1, label="avg regular price")
    ax.legend(fontsize=7)
    st.pyplot(fig)

# ---------------------------------------------------------------- decision
st.subheader(f"3-4. Forecast for week ending {next_week} and estimated elasticity")
m1, m2, m3 = st.columns(3)
m1.metric("Forecast units at current price", f"{Q0:,.0f}", help="XGBoost, bias-corrected, chosen promotion state")
if np.isnan(E):
    m2.metric("Elasticity", "not reliable")
    st.warning("No reliable elasticity for this product (see notebook 03). Recommendation: keep the current "
               "price and run a controlled price test.")
    st.stop()
m2.metric("Price elasticity (promotion-identified)", f"{E:.2f}", help=f"95% CI {E - 1.96 * E_se:.2f} to {E + 1.96 * E_se:.2f}; "
                                                         f"source: {source}")
m3.metric("Demand is", "elastic" if E < -1 else "inelastic")

cost = (1 - margin) * current_price
r = opt.optimize_price(Q0, current_price, E, unit_cost=cost, objective=objective,
                       max_change=max_change, max_volume_loss=vol)
checks = opt.sanity_checks(r) if objective == "profit" else None

st.subheader("5-6. Demand and objective vs price")
curve = opt.objective_curve(Q0, current_price, E, cost, span=max(max_change + 0.1, 0.3))
fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
for ax, col in zip(axes, ["demand", "revenue", "profit"]):
    ax.plot(curve["price"], curve[col])
    ax.axvspan(r["lower_bound"], r["upper_bound"], color="green", alpha=0.1, label="feasible")
    ax.axvline(current_price, ls="--", color="grey", label="current")
    ax.axvline(r["recommended_price"], color="red", label="recommended")
    ax.set_title(col + (" (assumed cost)" if col == "profit" else ""))
axes[0].legend(fontsize=7)
st.pyplot(fig)

st.subheader("7-9. Recommendation")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Recommended price", f"${r['recommended_price']:.2f}", f"{r['price_change_pct']:+.1%}")
k2.metric("Expected units", f"{r['demand_recommended']:,.0f}", f"{r['demand_recommended'] / Q0 - 1:+.1%}")
k3.metric("Expected revenue", f"${r['revenue_recommended']:,.0f}",
          f"{r['revenue_recommended'] / r['revenue_current'] - 1:+.1%}")
k4.metric("Expected profit (assumed cost)", f"${r['profit_recommended']:,.0f}",
          f"{r['profit_recommended'] / r['profit_current'] - 1:+.1%}")
if objective == "profit":
    st.info(opt.interpret(r, margin))
else:
    st.info(f"Revenue objective ignores cost. With constant elasticity E = {E:.2f}, revenue moves like P^(1+E): "
            + ("it rises with price (|E| < 1), so the optimizer goes to the highest allowed price."
               if E > -1 else "it falls with price (|E| > 1), so the optimizer goes to the lowest allowed price.")
            + f" Binding constraint: {r['binding_constraint']}.")

st.subheader("Dynamic price path")
target_m = r["recommended_price"] / current_price
path = opt.price_path(target_m, weeks=8)
p1, p2 = st.columns([1, 2])
p1.metric("Next week's price (max 5%/week)", f"${current_price * path[0]:.2f}", f"{path[0] - 1:+.1%}")
p1.metric("Weeks to reach target", opt.weeks_to_target(target_m))
with p2:
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.step(range(0, 9), [current_price] + list(current_price * path), where="post")
    ax.axhline(r["recommended_price"], color="red", ls="--", lw=1, label="target")
    ax.set(xlabel="weeks from now", ylabel="price ($)")
    ax.legend(fontsize=7)
    st.pyplot(fig)
st.caption("Elasticities are re-estimated every quarter in the pipeline; the target is recomputed each week from the latest forecast.")

st.subheader("10. Constraints")
st.table(pd.DataFrame({
    "constraint": ["price band", "volume guardrail", "cost floor", "weekly change limit", "binding constraint"],
    "value": [f"${r['lower_bound']:.2f} - ${current_price * (1 + max_change):.2f}",
              f"units >= {1 - vol:.0%} of forecast -> price <= ${r['upper_bound']:.2f}",
              f"price > ${cost:.2f} (ASSUMED)", f"<= {config.MAX_WEEKLY_CHANGE:.0%} per week",
              r["binding_constraint"]]}))
if checks is not None:
    st.caption("Sanity checks: " + ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in checks.items()))
st.caption("All expected values are model-based estimates under a constant-elasticity assumption, "
           "not observed outcomes. Validate with a controlled price test before rollout.")
