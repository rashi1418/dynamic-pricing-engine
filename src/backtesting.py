"""Offline counterfactual simulation (historical policy simulation).

This is NOT an observed experiment. We never see what customers would have bought
at the alternative price. What we do:

  1. Use ONLY training data (2009-2010) to estimate elasticities and choose, per
     product, a regular-price multiplier m* in [0.8, 1.2] (profit objective,
     ASSUMED unit cost, volume guardrail).
  2. Replay the 2011 test period. For every observed store-week we keep the
     retailer's actual promotion calendar and discount depth, but scale the
     shelf price by m*:
         P_sim = m* x P_obs
         Q_sim = Q_obs x (m*) ** E          (constant-elasticity adjustment)
  3. Compare simulated vs historical revenue / (assumed) profit.

Anchoring the simulation to observed demand Q_obs keeps the real demand shocks of
2011 (weather, competitors, promotions) in both scenarios, so the difference is
driven only by the price change and the assumed elasticity.

Because the result depends entirely on E, the same policy is re-evaluated under
alternative "true" elasticities (stress tests). A policy that only looks good
under its own elasticity is not trustworthy.
"""
import numpy as np
import pandas as pd

from . import config
from .optimization import optimize_price


def reference_prices(panel_train: pd.DataFrame, weeks=config.REFERENCE_WEEKS) -> pd.DataFrame:
    """Current regular price per product = demand-weighted mean base price over the
    last `weeks` weeks of the training period; unit cost is derived from it."""
    last = np.sort(panel_train["week"].unique())[-weeks:]
    d = panel_train[panel_train["week"].isin(last)]
    ref = d.groupby("upc").apply(lambda x: np.average(x["base_price"], weights=x["units"]),
                                 include_groups=False).rename("ref_price").reset_index()
    ref["unit_cost_assumed"] = (1 - config.ASSUMED_GROSS_MARGIN) * ref["ref_price"]
    return ref


def choose_multipliers(elast: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Profit-optimal multiplier per product (Q0 cancels for constant elasticity)."""
    rows = []
    for r in elast.merge(ref, on="upc").itertuples(index=False):
        if np.isnan(r.elasticity):
            rows.append({"upc": r.upc, "multiplier": 1.0, "elasticity": np.nan})
            continue
        o = optimize_price(Q0=1.0, P0=r.ref_price, E=r.elasticity, unit_cost=r.unit_cost_assumed)
        rows.append({"upc": r.upc, "multiplier": o["recommended_price"] / r.ref_price,
                     "elasticity": r.elasticity, "solution_type": o["solution_type"],
                     "binding_constraint": o["binding_constraint"]})
    return pd.DataFrame(rows)


def simulate(panel_test: pd.DataFrame, policy: pd.DataFrame, ref: pd.DataFrame,
             true_elasticity: pd.DataFrame | None = None) -> pd.DataFrame:
    """Replay the test period under `policy` (columns upc, multiplier).

    true_elasticity (columns upc, elasticity) = the elasticity assumed to describe
    real customer behaviour. Defaults to the policy's own elasticity.
    """
    d = panel_test[["upc", "week", "store_num", "price", "units"]].merge(
        policy[["upc", "multiplier", "elasticity"]], on="upc").merge(
        ref[["upc", "unit_cost_assumed"]], on="upc")
    if true_elasticity is not None:
        d = d.drop(columns="elasticity").merge(true_elasticity[["upc", "elasticity"]], on="upc", how="left")
    d["elasticity"] = d["elasticity"].fillna(0.0)
    d.loc[d["multiplier"] == 1.0, "elasticity"] = 0.0
    d["price_sim"] = d["price"] * d["multiplier"]
    d["units_sim"] = d["units"] * d["multiplier"] ** d["elasticity"]
    d["revenue_hist"] = d["price"] * d["units"]
    d["revenue_sim"] = d["price_sim"] * d["units_sim"]
    d["profit_hist"] = (d["price"] - d["unit_cost_assumed"]) * d["units"]
    d["profit_sim"] = (d["price_sim"] - d["unit_cost_assumed"]) * d["units_sim"]
    return d


def summarize(sim: pd.DataFrame, by="upc") -> pd.DataFrame:
    agg = sim.groupby(by).agg(
        multiplier=("multiplier", "first"),
        units_hist=("units", "sum"), units_sim=("units_sim", "sum"),
        revenue_hist=("revenue_hist", "sum"), revenue_sim=("revenue_sim", "sum"),
        profit_hist=("profit_hist", "sum"), profit_sim=("profit_sim", "sum"))
    agg["units_change_pct"] = agg["units_sim"] / agg["units_hist"] - 1
    agg["revenue_change_pct"] = agg["revenue_sim"] / agg["revenue_hist"] - 1
    # a % change is undefined when the base is <= 0 (e.g. clearance weeks with
    # negative ASSUMED profit), so report the absolute difference instead
    agg["profit_change_pct"] = np.where(agg["profit_hist"] > 0, agg["profit_sim"] / agg["profit_hist"] - 1, np.nan)
    agg["profit_change_abs"] = agg["profit_sim"] - agg["profit_hist"]
    return agg


def total_row(sim: pd.DataFrame, label: str) -> dict:
    t = sim[["units", "units_sim", "revenue_hist", "revenue_sim", "profit_hist", "profit_sim"]].sum()
    return {"scenario": label,
            "revenue_hist": t["revenue_hist"], "revenue_sim": t["revenue_sim"],
            "revenue_change_pct": t["revenue_sim"] / t["revenue_hist"] - 1,
            "profit_hist": t["profit_hist"], "profit_sim": t["profit_sim"],
            "profit_change_pct": t["profit_sim"] / t["profit_hist"] - 1,
            "units_change_pct": t["units_sim"] / t["units"] - 1}


# ==========================================================================
# 1. Price-response validation on held-out price changes
# ==========================================================================
def lift_events(panel: pd.DataFrame, start, end, n_base=8, min_base=3) -> pd.DataFrame:
    """Held-out temporary-price-reduction events with an observed sales lift.

    For each store-product-week in [start, end] with a pure price cut (discount > 2%,
    no feature ad, no display):
        observed lift   = log units - mean log units of the same store-product in its
                          previous `n_base` non-promoted weeks (past only)
        price change    = log price - mean log price in those baseline weeks
    'Within-week' versions subtract the same quantities averaged over the stores that
    did NOT run a promotion on that product that week, which removes chain-wide demand
    shocks (holidays, weather, national ads) from the comparison.
    """
    d = panel.sort_values("week").copy()
    d["clean_base"] = (d["any_promo"] == 0)
    key = ["store_num", "upc"]
    lu = d["log_units"].where(d["clean_base"])
    lp = d["log_price"].where(d["clean_base"])
    g = d.groupby(key)
    d["base_lu"] = lu.groupby([d.store_num, d.upc]).transform(
        lambda s: s.shift(1).rolling(n_base * 3, min_periods=1).apply(lambda x: np.nanmean(x[~np.isnan(x)][-n_base:]) if np.sum(~np.isnan(x)) >= min_base else np.nan, raw=True))
    d["base_lp"] = lp.groupby([d.store_num, d.upc]).transform(
        lambda s: s.shift(1).rolling(n_base * 3, min_periods=1).apply(lambda x: np.nanmean(x[~np.isnan(x)][-n_base:]) if np.sum(~np.isnan(x)) >= min_base else np.nan, raw=True))
    d["obs_lift"] = d["log_units"] - d["base_lu"]
    d["dlogp"] = d["log_price"] - d["base_lp"]
    d = d[(d["week"] >= start) & (d["week"] <= end)].dropna(subset=["obs_lift", "dlogp"])
    ctrl = d[d["clean_base"]].groupby(["upc", "week"]).agg(ctrl_lift=("obs_lift", "mean"), ctrl_dlogp=("dlogp", "mean"))
    ev = d[(d["discount_pct"] > 0.02) & (d["feature"] == 0) & (d["display"] == 0)].join(ctrl, on=["upc", "week"])
    ev = ev.dropna(subset=["ctrl_lift"])
    ev["obs_lift_ww"] = ev["obs_lift"] - ev["ctrl_lift"]
    ev["dlogp_ww"] = ev["dlogp"] - ev["ctrl_dlogp"]
    return ev[["week", "store_num", "upc", "store_segment", "description", "category", "discount_pct",
               "obs_lift", "dlogp", "obs_lift_ww", "dlogp_ww"]]


def score_price_response(events: pd.DataFrame, elasticities: dict) -> pd.DataFrame:
    """Score each elasticity model on held-out lift events.

    elasticities: {model_name: DataFrame(upc, [store_segment], elasticity)}
    Metrics (on the within-week version, plus the simple version for reference):
      MAE              mean |observed lift - predicted lift| (log points)
      MAE skill        1 - MAE / MAE of the 'no price effect' model (> 0 = better than ignoring price)
      calibration      slope of observed on predicted lift (1 = right magnitude,
                       < 1 = model over-states the response, > 1 = under-states)
    """
    rows = []
    zero_ww = np.mean(np.abs(events["obs_lift_ww"]))
    zero_s = np.mean(np.abs(events["obs_lift"]))
    rows.append({"model": "No price effect (E = 0)", "MAE_within_week": zero_ww, "MAE_skill_within_week": 0.0,
                 "calibration_slope": np.nan, "MAE_simple": zero_s, "MAE_skill_simple": 0.0, "n_events": len(events)})
    for name, e in elasticities.items():
        on = ["upc", "store_segment"] if "store_segment" in e.columns else ["upc"]
        x = events.merge(e[on + ["elasticity"]], on=on, how="inner").dropna(subset=["elasticity"])
        pred_ww, pred_s = x["elasticity"] * x["dlogp_ww"], x["elasticity"] * x["dlogp"]
        mae_ww = np.mean(np.abs(x["obs_lift_ww"] - pred_ww))
        mae_s = np.mean(np.abs(x["obs_lift"] - pred_s))
        z_ww = np.mean(np.abs(x["obs_lift_ww"]))
        z_s = np.mean(np.abs(x["obs_lift"]))
        slope = np.polyfit(pred_ww, x["obs_lift_ww"], 1)[0] if pred_ww.std() > 0 else np.nan
        rows.append({"model": name, "MAE_within_week": mae_ww, "MAE_skill_within_week": 1 - mae_ww / z_ww,
                     "calibration_slope": slope, "MAE_simple": mae_s, "MAE_skill_simple": 1 - mae_s / z_s,
                     "n_events": len(x)})
    return pd.DataFrame(rows)


# ==========================================================================
# 2. Walk-forward backtest of the dynamic pricing engine
# ==========================================================================
def segment_reference_prices(panel_hist: pd.DataFrame, weeks=config.REFERENCE_WEEKS) -> pd.DataFrame:
    last = np.sort(panel_hist["week"].unique())[-weeks:]
    d = panel_hist[panel_hist["week"].isin(last)]
    return d.groupby(["upc", "store_segment"]).apply(
        lambda x: np.average(x["base_price"], weights=x["units"]), include_groups=False).rename("ref_price").reset_index()


def segment_targets(seg_el: pd.DataFrame, ref_seg: pd.DataFrame, cost: pd.DataFrame) -> pd.DataFrame:
    """Profit-optimal target multiplier per product x store segment."""
    from .optimization import optimize_price
    x = seg_el.rename(columns={"segment": "store_segment"}).merge(ref_seg, on=["upc", "store_segment"]) \
        .merge(cost[["upc", "unit_cost_assumed"]], on="upc")
    rows = []
    for r in x.itertuples(index=False):
        if not r.usable or np.isnan(r.segment_elasticity):
            rows.append({"upc": r.upc, "store_segment": r.store_segment, "target": 1.0,
                         "elasticity": r.segment_elasticity, "binding_constraint": "no usable elasticity"})
            continue
        o = optimize_price(1.0, r.ref_price, r.segment_elasticity, unit_cost=r.unit_cost_assumed)
        rows.append({"upc": r.upc, "store_segment": r.store_segment, "target": o["recommended_price"] / r.ref_price,
                     "elasticity": r.segment_elasticity, "binding_constraint": o["binding_constraint"]})
    return pd.DataFrame(rows)


def dynamic_schedule(panel: pd.DataFrame, test_weeks, estimate_fn, cost: pd.DataFrame,
                     every=config.REESTIMATE_EVERY_WEEKS, max_step=config.MAX_WEEKLY_CHANGE):
    """Weekly multiplier for every product x segment over `test_weeks`.

    At each re-estimation date only data BEFORE that date is used. Between dates the
    multiplier moves toward the latest target by at most `max_step` per week.
    estimate_fn(history_panel) -> segment elasticity table (decision_elasticities()[1]).
    """
    test_weeks = list(test_weeks)
    sched, refits = [], []
    current = {}
    targets, el_now = None, None
    for i, wk in enumerate(test_weeks):
        if i % every == 0:
            hist = panel[panel["week"] < wk]
            el_now = estimate_fn(hist)
            targets = segment_targets(el_now, segment_reference_prices(hist), cost)
            refits.append(targets.assign(refit_week=wk))
        for r in targets.itertuples(index=False):
            k = (r.upc, r.store_segment)
            m = current.get(k, 1.0)
            if r.target > m:
                m = min(r.target, m * (1 + max_step))
            elif r.target < m:
                m = max(r.target, m * (1 - max_step))
            current[k] = m
            sched.append({"week": wk, "upc": r.upc, "store_segment": r.store_segment, "multiplier": m,
                          "elasticity": r.elasticity})
    return pd.DataFrame(sched), pd.concat(refits, ignore_index=True)


def simulate_dynamic(panel_test: pd.DataFrame, schedule: pd.DataFrame, cost: pd.DataFrame,
                     true_elasticity: pd.DataFrame | None = None) -> pd.DataFrame:
    """Replay the test period under a weekly schedule (same logic as simulate())."""
    d = panel_test[["upc", "week", "store_num", "store_segment", "price", "units"]].merge(
        schedule, on=["week", "upc", "store_segment"], how="left").merge(cost[["upc", "unit_cost_assumed"]], on="upc")
    d["multiplier"] = d["multiplier"].fillna(1.0)
    if true_elasticity is not None:
        on = ["upc", "store_segment"] if "store_segment" in true_elasticity.columns else ["upc"]
        d = d.drop(columns="elasticity").merge(true_elasticity[on + ["elasticity"]], on=on, how="left")
    d["elasticity"] = d["elasticity"].fillna(0.0)
    d.loc[d["multiplier"] == 1.0, "elasticity"] = 0.0
    d["price_sim"] = d["price"] * d["multiplier"]
    d["units_sim"] = d["units"] * d["multiplier"] ** d["elasticity"]
    d["revenue_hist"] = d["price"] * d["units"]
    d["revenue_sim"] = d["price_sim"] * d["units_sim"]
    d["profit_hist"] = (d["price"] - d["unit_cost_assumed"]) * d["units"]
    d["profit_sim"] = (d["price_sim"] - d["unit_cost_assumed"]) * d["units_sim"]
    return d
