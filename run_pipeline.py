"""Run the whole project end to end.

    python run_pipeline.py --download          # fetch + verify the raw workbook, then run all stages
    python run_pipeline.py                     # run all stages (raw file already in data/raw/)
    python run_pipeline.py --stage elasticity  # run a single stage

Stages (each reads the outputs of the previous ones from disk):
    data -> features -> cv -> forecast -> shap -> elasticity -> optimize -> backtest
All tables go to reports/results/, figures to reports/figures/, models to models/.
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import (backtesting as bt, config, data_loader, elasticity as el, feature_engineering as fe,
                 forecasting as fc, optimization as opt, preprocessing as pp, visualization as viz)

R = config.RESULTS_DIR
np.random.seed(config.SEED)


def _metrics_store(update: dict):
    path = R / "metrics.json"
    m = json.loads(path.read_text()) if path.exists() else {}
    m.update(update)
    path.write_text(json.dumps(m, indent=2, default=float))


def load_features():
    return fe.apply_category_levels(pd.read_parquet(config.FEATURES_FILE))


# ------------------------------------------------------------------ stages
def stage_data():
    tx, stores, products = data_loader.load_raw()
    panel, log = pp.clean_transactions(tx, stores, products)
    panel.to_parquet(config.PANEL_FILE, index=False)
    log.to_csv(R / "cleaning_log.csv", index=False)
    pp.coverage_table(panel).to_csv(R / "coverage_by_product.csv", index=False)
    viz.weekly_units_by_category(panel)
    viz.distributions(panel)
    _, lift = viz.promo_lift(panel)
    lift.to_csv(R / "promo_lift_by_category.csv")
    _metrics_store({"data": {
        "raw_rows": int(len(tx)), "clean_rows": int(len(panel)),
        "weeks": int(panel.week.nunique()), "stores": int(panel.store_num.nunique()),
        "products": int(panel.upc.nunique()), "first_week": str(panel.week.min().date()),
        "last_week": str(panel.week.max().date()),
        "share_rows_discounted": float(panel.on_discount.mean()),
        "share_rows_any_promo": float(panel.any_promo.mean())}})
    print(f"clean panel: {panel.shape}")


def stage_features():
    panel = pd.read_parquet(config.PANEL_FILE)
    feats = fe.build_feature_table(panel)
    feats.to_parquet(config.FEATURES_FILE, index=False)
    fe.save_category_levels(feats)
    obs = feats["observed"].mean()
    _metrics_store({"features": {"grid_rows": int(len(feats)), "share_grid_weeks_observed": float(obs),
                                 "n_features": len(fe.feature_columns())}})
    print(f"feature table: {feats.shape}, observed share {obs:.3f}")


def stage_cv(folds=None):
    """Walk-forward CV, one fold at a time (checkpointed, each fold ~1 min)."""
    df = load_features()
    for k in folds or range(1, config.N_CV_FOLDS + 1):
        res, info = fc.run_cv_fold(df, k)
        res.to_csv(R / f"cv_fold{k}.csv", index=False)
        (R / f"cv_fold{k}.json").write_text(json.dumps(info))
        print(res[["model", "WMAPE"]].to_string(index=False))


def stage_forecast():
    df = load_features()
    ks = range(1, config.N_CV_FOLDS + 1)
    cv = pd.concat([pd.read_csv(R / f"cv_fold{k}.csv") for k in ks], ignore_index=True)
    info = fc.combine_cv([json.loads((R / f"cv_fold{k}.json").read_text()) for k in ks])
    cv.to_csv(R / "cv_results.csv", index=False)
    models, table, preds = fc.fit_final_and_test(df, info)
    table.to_csv(R / "model_comparison_test.csv")
    config.MODELS_DIR.mkdir(exist_ok=True)
    models["XGBoost"].save_model(config.MODELS_DIR / "xgb_train_only.json")
    models["LightGBM"].booster_.save_model(str(config.MODELS_DIR / "lgb_train_only.txt"))
    _, te = fc.train_test_masks(df)
    out = df.loc[te, ["week", "store_num", "upc", "description", "units"]].copy()
    for k, v in preds.items():
        out[k] = v
    out.to_parquet(R / "test_predictions.parquet", index=False)
    viz.model_comparison(table.drop(index=[i for i in table.index if "no bias" in i]))
    viz.actual_vs_forecast(df.loc[te], {k: preds[k] for k in ["Seasonal naive (lag 52)", "Naive (last week)", "XGBoost"]})
    _metrics_store({"forecast": {"cv_info": info, "test": table.to_dict(orient="index"),
                                 "cv_mean": cv.groupby("model")[["MAE", "RMSE", "WMAPE"]].mean().to_dict(orient="index")}})
    print(table.round(4).to_string())


def stage_shap():
    import shap
    import xgboost as xgb
    df = load_features()
    _, te = fc.train_test_masks(df)
    model = xgb.XGBRegressor()
    model.load_model(config.MODELS_DIR / "xgb_train_only.json")
    X = df.loc[te, fe.feature_columns()].sample(3000, random_state=config.SEED)
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv.values).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(R / "shap_importance.csv", index=False)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=20, show=False)
    plt.title("SHAP summary (XGBoost, log1p units scale, 2011 sample)")
    plt.savefig(config.FIG_DIR / "shap_summary.png", dpi=110, bbox_inches="tight")
    plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, f in zip(axes, ["price_vs_recent_base", "discount_pct", "units_rollmean_4"]):
        shap.plots.scatter(sv[:, f], ax=ax, show=False)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "shap_dependence.png", dpi=110, bbox_inches="tight")
    plt.close("all")
    print(imp.head(15).to_string())


def _decision_tables(panel, var, with_iv=True):
    """Promotion-identified estimates -> (product table, product decision, segment decision)."""
    prod, seg = el.estimate_promo_identified(panel, var, with_iv=with_iv)
    pdec, sdec = el.decision_elasticities(prod, seg)
    return prod, pdec, sdec


def _decision_file(pdec):
    """decision_elasticity.csv: one product-level elasticity per product (NaN = not usable)."""
    d = pdec.copy()
    d.loc[~d["usable"], ["elasticity", "se"]] = np.nan
    return d


def stage_elasticity():
    panel = pd.read_parquet(config.PANEL_FILE)
    hol = el.holiday_series(panel)
    var = el.price_variation_table(panel)
    var.to_csv(R / "price_variation.csv", index=False)
    # teaching ladder: how controls change the naive estimate
    ladder = el.estimate_all_products(panel, var, hol)
    ladder.to_csv(R / "elasticity_ladder.csv", index=False)
    robust = el.estimate_all_products(panel, var, hol, exclude_outliers=True)
    robust.to_csv(R / "elasticity_ladder_no_outliers.csv", index=False)
    pooled = el.estimate_category_pooled(panel)
    pooled.to_csv(R / "elasticity_pooled.csv", index=False)
    # decision model: promotion-identified (M8) + shrinkage + store segments
    prod, pdec, sdec = _decision_tables(panel, var)
    prod.to_csv(R / "promo_identified_elasticities.csv", index=False)
    pdec.to_csv(R / "product_elasticity_shrinkage.csv", index=False)
    sdec.to_csv(R / "segment_elasticity.csv", index=False)
    sel = _decision_file(pdec)
    sel.to_csv(R / "decision_elasticity.csv", index=False)
    viz.decision_elasticities(sel.assign(source=np.where(sel["elasticity"].notna(), "product", "none")))
    wide = ladder.pivot_table(index="upc", columns="spec", values="elasticity")
    changed = wide[(wide["M1 naive"] > 0) & (wide["M4 + store FE + week FE"] < 0)].index.tolist()
    show = changed + [u for u in ladder[ladder.spec == "M4 + store FE + week FE"].sort_values("elasticity")["upc"].head(3)]
    viz.elasticity_ladder(ladder, show[:8])
    for u in changed[:1]:
        viz.confounding_scatter(panel, u, name="el_confounding_example")
    iv = prod[prod["spec"] == "M7 promotion-calendar IV"]
    m8 = prod[prod["spec"] == el.DECISION_SPEC]
    _metrics_store({"elasticity": {
        "n_eligible": int(var.eligible.sum()),
        "median_by_spec": {**ladder.groupby("spec")["elasticity"].median().to_dict(),
                           **prod.groupby("spec")["elasticity"].median().to_dict()},
        "share_negative_by_spec": {**ladder.groupby("spec")["elasticity"].apply(lambda s: (s < 0).mean()).to_dict(),
                                   **prod.groupby("spec")["elasticity"].apply(lambda s: (s < 0).mean()).to_dict()},
        "iv_first_stage_F_median": float(iv["first_stage_F"].median()),
        "iv_weak_instruments": int((iv["first_stage_F"] < 10).sum()),
        "m8_median_raw": float(m8["elasticity"].median()),
        "decision_median_shrunk": float(sel["elasticity"].median()),
        "n_usable": int(pdec["usable"].sum()), "n_not_usable": int((~pdec["usable"]).sum()),
        "segment_deviations_significant": int((sdec["deviation_se"] > 0).sum() and
                                              ((sdec["deviation"] / sdec["deviation_se"].replace(0, np.nan)).abs() > 2).sum()),
        "segment_deviations_total": int((sdec["segment"] != "MAINSTREAM").sum()),
        "n_sign_flip_naive_positive_to_controlled_negative": len(changed)}})
    print(pdec.describe())


def stage_optimize():
    """Dynamic pricing engine for the next (unobserved) week, using ALL data:
    target price per product x store segment + the first step of the gradual price path."""
    import xgboost as xgb
    panel = pd.read_parquet(config.PANEL_FILE)
    info = json.loads((R / "metrics.json").read_text())["forecast"]["cv_info"]
    next_week = panel["week"].max() + pd.Timedelta(days=7)
    feats = fe.apply_category_levels(fe.build_feature_table(panel, future_week=next_week))
    model = fc.fit_production_model(feats, info["n_estimators"]["XGBoost"])
    model.save_model(config.MODELS_DIR / "xgb_production.json")
    calib = info["bias_correction"]["XGBoost"]

    fut = feats[feats["week"] == next_week].copy()
    # current regular price = the store-product's most recent base price
    last = panel.sort_values("week").groupby(["store_num", "upc"]).tail(1)[["store_num", "upc", "base_price"]]
    fut = fut.drop(columns=["base_price"]).merge(last, on=["store_num", "upc"], how="left")
    fut = fe.apply_category_levels(fut)
    fut = fe.set_price_scenario(fut, price=fut["base_price"], base_price=fut["base_price"])
    fut["baseline_units"] = fc.predict_units(model, fut) * calib
    fut.to_parquet(config.DATA_PROCESSED / "next_week_features.parquet", index=False)
    _metrics_store({"production": {"next_week": str(next_week.date()), "bias_correction": calib}})

    sel = pd.read_csv(R / "decision_elasticity.csv")
    segs = pd.read_csv(R / "segment_elasticity.csv")
    names = panel.groupby("upc").apply(lambda d: f"{d['description'].iloc[0]} {d['product_size'].iloc[0]}",
                                       include_groups=False).rename("product")

    # distribution check: is the product still on the shelf?
    wk = panel.groupby(["upc", "week"])["store_num"].nunique().rename("stores").reset_index()
    last4 = np.sort(panel["week"].unique())[-4:]
    dist = wk.groupby("upc").agg(usual_stores=("stores", lambda s: s.quantile(0.9)), last_sale=("week", "max"))
    dist["stores_last_4_weeks"] = wk[wk["week"].isin(last4)].groupby("upc")["stores"].sum() / len(last4)
    dist["stores_last_4_weeks"] = dist["stores_last_4_weeks"].fillna(0)
    dist["current_distribution"] = dist["stores_last_4_weeks"] / dist["usual_stores"]
    skip = dist[dist["current_distribution"] < config.MIN_CURRENT_DISTRIBUTION].join(names)
    skip["reason"] = np.where(skip["stores_last_4_weeks"] == 0,
                              "discontinued: no sales since " + skip["last_sale"].dt.date.astype(str),
                              "being phased out: sold in ~" + skip["stores_last_4_weeks"].round(1).astype(str)
                              + " stores/week over the last 4 weeks vs ~" + skip["usual_stores"].round().astype(int).astype(str)
                              + " normally")
    no_el = sel[sel["elasticity"].isna() & ~sel["upc"].isin(skip.index)].join(names, on="upc")
    no_el = no_el.assign(reason="no usable price response: promotion-identified elasticity (after shrinkage) "
                                "is not clearly negative - keep current price and test")
    skipped = pd.concat([skip.reset_index()[["upc", "product", "reason"]], no_el[["upc", "product", "reason"]]])
    skipped.to_csv(R / "not_recommended_products.csv", index=False)
    fut = fut[~fut["upc"].isin(skipped["upc"])]

    # product-level unit cost (ASSUMED) from the chain demand-weighted current price
    chain = fut.groupby("upc").apply(lambda d: pd.Series({
        "P0_chain": np.average(d["base_price"], weights=d["baseline_units"]),
        "Q0_chain": d["baseline_units"].sum()}), include_groups=False)
    chain["unit_cost_assumed"] = (1 - config.ASSUMED_GROSS_MARGIN) * chain["P0_chain"]
    fut["segment_key"] = fut["store_segment"].astype(str)  # grouping key; model input stays categorical
    base = fut.groupby(["upc", "segment_key"]).apply(lambda d: pd.Series({
        "stores": d["store_num"].nunique(), "Q0": d["baseline_units"].sum(),
        "P0": np.average(d["base_price"], weights=d["baseline_units"])}), include_groups=False).reset_index() \
        .rename(columns={"segment_key": "store_segment"})
    base = base.merge(segs.rename(columns={"segment": "store_segment"})[["upc", "store_segment", "segment_elasticity"]],
                      on=["upc", "store_segment"], how="left").join(chain, on="upc").join(names, on="upc")
    base = base.merge(sel[["upc", "category", "se"]], on="upc")

    rows, checks = [], []
    for r in base.itertuples(index=False):
        E = r.segment_elasticity
        o = opt.optimize_price(r.Q0, r.P0, E, unit_cost=r.unit_cost_assumed)
        c = opt.sanity_checks(o)
        target_m = o["recommended_price"] / r.P0
        step_m = opt.price_path(target_m, weeks=1)[0]
        rows.append({"upc": r.upc, "product": r.product, "category": r.category, "store_segment": r.store_segment,
                     "stores": r.stores, **o, "next_week_price": r.P0 * step_m,
                     "next_week_change_pct": step_m - 1, "weeks_to_target": opt.weeks_to_target(target_m),
                     "explanation": opt.interpret(o)})
        checks.append({"upc": r.upc, "product": r.product, "store_segment": r.store_segment, **c})
    rec = pd.DataFrame(rows)
    rec.to_csv(R / "recommendations_full.csv", index=False)
    pd.DataFrame(checks).to_csv(R / "optimization_sanity_checks.csv", index=False)
    business = rec[["product", "category", "store_segment", "stores", "current_price", "recommended_price",
                    "price_change_pct", "next_week_price", "next_week_change_pct", "weeks_to_target", "elasticity",
                    "demand_current", "demand_recommended", "revenue_current", "revenue_recommended",
                    "profit_current", "profit_recommended", "binding_constraint"]].rename(columns={
                        "recommended_price": "target_price", "price_change_pct": "target_change_pct",
                        "demand_current": "current_weekly_units", "demand_recommended": "expected_weekly_units_at_target",
                        "revenue_current": "current_weekly_revenue", "revenue_recommended": "expected_weekly_revenue_at_target",
                        "profit_current": "current_weekly_profit_ASSUMED_COST",
                        "profit_recommended": "expected_weekly_profit_at_target_ASSUMED_COST"})
    business.sort_values(["category", "product", "store_segment"]).to_csv(R / "business_recommendations.csv", index=False)
    by_prod = rec.groupby(["upc", "product", "category"]).apply(lambda d: pd.Series({
        "current_price": np.average(d["current_price"], weights=d["demand_current"]),
        "target_price": np.average(d["recommended_price"], weights=d["demand_current"]),
        "elasticity_range": f"{d['elasticity'].min():.2f} to {d['elasticity'].max():.2f}",
        "current_weekly_units": d["demand_current"].sum(), "expected_weekly_units": d["demand_recommended"].sum(),
        "current_weekly_revenue": d["revenue_current"].sum(), "expected_weekly_revenue": d["revenue_recommended"].sum(),
        "current_weekly_profit_ASSUMED_COST": d["profit_current"].sum(),
        "expected_weekly_profit_ASSUMED_COST": d["profit_recommended"].sum()}), include_groups=False).reset_index()
    by_prod["target_change_pct"] = by_prod["target_price"] / by_prod["current_price"] - 1
    by_prod.sort_values("target_change_pct").to_csv(R / "business_recommendations_by_product.csv", index=False)

    for tag, idx in (("most_elastic", rec["elasticity"].idxmin()), ("least_elastic", rec["elasticity"].idxmax()),
                     ("median", (rec["elasticity"] - rec["elasticity"].median()).abs().idxmin())):
        viz.objective_curves(rec.loc[idx].to_dict(), title=f"{rec.loc[idx, 'product']} - {rec.loc[idx, 'store_segment']} stores",
                             name=f"opt_curves_{tag}")
    # diagnostic: what does XGBoost itself imply about regular price?
    resp = []
    for u in sel.dropna(subset=["elasticity"])["upc"]:
        if u not in set(fut["upc"]):
            continue
        d = fut[fut["upc"] == u]
        mr = opt.model_price_response(model, d)
        lo, hi = mr.iloc[0]["mean_pred_units"], mr.iloc[-1]["mean_pred_units"]
        resp.append({"upc": u, "xgb_arc_elasticity_0.8_to_1.2": np.log(hi / lo) / np.log(1.2 / 0.8),
                     "xgb_monotone_decreasing": bool(np.all(np.diff(mr["mean_pred_units"]) <= 1e-9))})
    resp = pd.DataFrame(resp).merge(sel[["upc", "elasticity"]], on="upc").join(names, on="upc")
    resp.to_csv(R / "xgb_price_response_diagnostic.csv", index=False)
    _metrics_store({"optimization": {
        "n_products": int(rec["upc"].nunique()), "n_product_segments": len(rec),
        "n_not_recommended": len(skipped), "binding": rec["binding_constraint"].value_counts().to_dict(),
        "median_target_change": float(rec["price_change_pct"].median()),
        "median_weeks_to_target": float(rec["weeks_to_target"].median()),
        "all_sanity_checks_passed": bool(pd.DataFrame(checks)["all_passed"].all()),
        "total_weekly_revenue_current": float(rec["revenue_current"].sum()),
        "total_weekly_revenue_at_target": float(rec["revenue_recommended"].sum()),
        "total_weekly_profit_current_assumed": float(rec["profit_current"].sum()),
        "total_weekly_profit_at_target_assumed": float(rec["profit_recommended"].sum()),
        "xgb_price_response_monotone_share": float(resp["xgb_monotone_decreasing"].mean()),
        "xgb_price_response_median_arc_elasticity": float(resp["xgb_arc_elasticity_0.8_to_1.2"].median())}})
    print(by_prod.round(3).to_string())


def stage_backtest():
    """Historical backtesting, everything estimated on 2009-2010 only:
    (1) price-response validation against real held-out 2011 price changes;
    (2) walk-forward simulation of the dynamic pricing engine over 2011."""
    panel = pd.read_parquet(config.PANEL_FILE)
    tr = panel[panel["week"] <= config.TRAIN_END]
    te = panel[panel["week"] >= config.TEST_START]

    # ---- (1) price-response validation -------------------------------------------
    var = el.price_variation_table(tr)
    ladder = el.estimate_all_products(tr, var, el.holiday_series(tr))
    prod, pdec, sdec = _decision_tables(tr, var)
    _decision_file(pdec).to_csv(R / "backtest_train_elasticity.csv", index=False)
    events = bt.lift_events(panel, config.TEST_START, panel["week"].max())
    pick = lambda df, spec: df[df["spec"] == spec][["upc", "elasticity"]]
    models = {
        "M1 naive OLS": pick(ladder, "M1 naive"),
        "M4 controlled OLS (store + week FE)": pick(ladder, "M4 + store FE + week FE"),
        "M5 regular-price OLS": pick(ladder, "M5 regular-price elasticity"),
        "M7 promotion-calendar IV": pick(prod, "M7 promotion-calendar IV"),
        "M8 promotion-identified (raw)": pick(prod, el.DECISION_SPEC),
        "M8 + shrinkage (product level)": pdec[["upc", "elasticity"]],
        "M8 + shrinkage + store segment (ENGINE)": sdec.rename(columns={"segment": "store_segment"})
            .assign(elasticity=sdec["segment_elasticity"].values)[["upc", "store_segment", "elasticity"]],
    }
    val = bt.score_price_response(events, models)
    val.to_csv(R / "price_response_validation.csv", index=False)
    viz.price_response_validation(val)

    # ---- (2) walk-forward dynamic engine -----------------------------------------
    cache = {}

    def estimate(hist):
        k = hist["week"].max()
        if k not in cache:
            v = el.price_variation_table(hist)
            cache[k] = _decision_tables(hist, v, with_iv=False)[2]
        return cache[k]

    test_weeks = sorted(te["week"].unique())
    cost = bt.reference_prices(tr)
    sched, refits = bt.dynamic_schedule(panel, test_weeks, estimate, cost)
    sched.to_parquet(R / "dynamic_price_schedule.parquet", index=False)
    refits.to_csv(R / "dynamic_refit_targets.csv", index=False)

    # alternative "truths" for stress tests
    seg_truth_2011 = _decision_tables(te, el.price_variation_table(te), with_iv=False)[2] \
        .rename(columns={"segment": "store_segment"}).assign(elasticity=lambda d: d["segment_elasticity"])
    seg_train = sdec.rename(columns={"segment": "store_segment"}).assign(elasticity=lambda d: d["segment_elasticity"])
    se_map = pdec.set_index("upc")["se"]
    less = seg_train.assign(elasticity=lambda d: np.minimum(d["elasticity"] + 1.96 * d["upc"].map(se_map), -0.05))
    more = seg_train.assign(elasticity=lambda d: d["elasticity"] - 1.96 * d["upc"].map(se_map))
    m4_truth = pick(ladder, "M4 + store FE + week FE")

    scen = []
    sim_main = bt.simulate_dynamic(te, sched, cost)
    scen.append(bt.total_row(sim_main, "Dynamic engine | judged by its own (quarterly re-estimated) E  [main]"))
    scen.append(bt.total_row(bt.simulate_dynamic(te, sched, cost, seg_truth_2011), "Dynamic engine | E re-estimated on 2011 data"))
    scen.append(bt.total_row(bt.simulate_dynamic(te, sched, cost, less), "Dynamic engine | E at less-elastic CI bound"))
    scen.append(bt.total_row(bt.simulate_dynamic(te, sched, cost, more), "Dynamic engine | E at more-elastic CI bound"))
    scen.append(bt.total_row(bt.simulate_dynamic(te, sched, cost, m4_truth), "Dynamic engine | judged by controlled OLS (M4) E"))

    # static policy: one product-level multiplier chosen on 2010-12-29, applied at once
    stat_el = _decision_file(pdec)[["upc", "elasticity"]]
    static = bt.choose_multipliers(stat_el, cost)
    scen.append(bt.total_row(bt.simulate(te, static, cost, seg_train[["upc", "store_segment", "elasticity"]]
                                         .groupby("upc", as_index=False)["elasticity"].mean()),
                             "Static policy (one price change, no segments)"))
    flat = static.assign(multiplier=1.05, elasticity=static["upc"].map(stat_el.set_index("upc")["elasticity"]).fillna(0))
    scen.append(bt.total_row(bt.simulate(te, flat, cost, stat_el.fillna(0)), "Rule of thumb: +5% on everything"))
    m1 = pick(ladder, "M1 naive")
    naive_sel = m1.assign(elasticity=lambda d: d["elasticity"].where(d["elasticity"] < 0))
    naive_policy = bt.choose_multipliers(naive_sel, cost)
    scen.append(bt.total_row(bt.simulate(te, naive_policy, cost, stat_el.fillna(0)), "Naive-OLS policy | judged by engine E"))
    for m in (0.20, 0.40):
        old = config.ASSUMED_GROSS_MARGIN
        config.ASSUMED_GROSS_MARGIN = m
        cost_m = bt.reference_prices(tr)
        sched_m, _ = bt.dynamic_schedule(panel, test_weeks, estimate, cost_m)
        scen.append(bt.total_row(bt.simulate_dynamic(te, sched_m, cost_m), f"Dynamic engine | assumed margin {m:.0%}"))
        config.ASSUMED_GROSS_MARGIN = old

    summary = pd.DataFrame(scen)
    summary.to_csv(R / "backtest_summary.csv", index=False)
    per_product = bt.summarize(sim_main).join(te.groupby("upc")["description"].first())
    per_product.to_csv(R / "backtest_by_product.csv")
    viz.backtest_scenarios(summary)
    viz.dynamic_paths(sched, te)
    _metrics_store({"backtest": summary.set_index("scenario").to_dict(orient="index"),
                    "price_response_validation": val.set_index("model").to_dict(orient="index"),
                    "dynamic": {"n_refits": int(refits["refit_week"].nunique()),
                                "median_target_by_refit": refits.groupby("refit_week")["target"].median()
                                    .rename(lambda d: str(pd.Timestamp(d).date())).to_dict(),
                                "binding": refits["binding_constraint"].value_counts().to_dict(),
                                "n_events_validation": int(len(events))}})
    print(val.round(4).to_string())
    print(summary.round(4).to_string())


STAGES = {"data": stage_data, "features": stage_features, "cv": stage_cv, "forecast": stage_forecast, "shap": stage_shap,
          "elasticity": stage_elasticity, "optimize": stage_optimize, "backtest": stage_backtest}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download + verify the raw data first")
    ap.add_argument("--stage", choices=list(STAGES), help="run only this stage")
    ap.add_argument("--fold", type=int, help="with --stage cv: run only this walk-forward fold")
    args = ap.parse_args()
    for d in (config.DATA_PROCESSED, config.RESULTS_DIR, config.FIG_DIR, config.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if args.download:
        data_loader.download_raw()
    for name, fn in STAGES.items():
        if args.stage and name != args.stage:
            continue
        t = time.time()
        print(f"\n===== stage: {name} =====")
        fn([args.fold] if args.fold else None) if name == "cv" else fn()
        print(f"[{name}] done in {time.time() - t:.0f}s")
