"""Plotting helpers. Every function returns the figure and optionally saves it."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name):
    if name:
        config.FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIG_DIR / f"{name}.png", dpi=110, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------- EDA
def weekly_units_by_category(panel, name="eda_weekly_units_by_category"):
    w = panel.groupby(["week", "category"])["units"].sum().unstack()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    w.plot(ax=ax, lw=1.4)
    for d in config.SUPER_BOWL_DATES:
        ax.axvline(d, color="grey", ls=":", lw=1)
    ax.axvline(config.TEST_START, color="black", ls="--", lw=1)
    ax.text(config.TEST_START, ax.get_ylim()[1] * 0.95, "  test period", fontsize=9)
    ax.set(title="Weekly units by category (dotted = Super Bowl weeks)", xlabel="", ylabel="units / week")
    return _save(fig, name)


def distributions(panel, name="eda_distributions"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(panel["units"], bins=100, ax=axes[0], log_scale=(True, False))
    axes[0].set(title="Units per store-week (log x-axis)", xlabel="units")
    sns.boxplot(data=panel, x="category", y="price", ax=axes[1])
    axes[1].set(title="Shelf price by category", xlabel="")
    axes[1].tick_params(axis="x", rotation=20)
    sns.histplot(panel.loc[panel["discount_pct"] > 0, "discount_pct"], bins=40, ax=axes[2])
    axes[2].set(title="Discount depth when discounted", xlabel="discount vs base price")
    fig.tight_layout()
    return _save(fig, name)


def promo_lift(panel, name="eda_promo_lift"):
    d = panel.copy()
    d["promo_type"] = np.select(
        [(d.feature == 1) & (d.display == 1), d.feature == 1, d.display == 1, d.on_discount == 1],
        ["feature+display", "feature", "display", "discount only"], "no promotion")
    # index units against the same store-product's no-promo average
    base = d[d.promo_type == "no promotion"].groupby(["store_num", "upc"])["units"].mean().rename("base")
    d = d.join(base, on=["store_num", "upc"])
    d["lift"] = d["units"] / d["base"]
    order = ["no promotion", "discount only", "display", "feature", "feature+display"]
    t = d.groupby(["category", "promo_type"])["lift"].median().unstack()[order]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    t.plot.bar(ax=ax)
    ax.set(title="Median units relative to the same store-product's non-promo average",
           ylabel="units / non-promo average", xlabel="")
    ax.axhline(1, color="black", lw=0.8)
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, name), t


def confounding_scatter(panel, upc, name=None):
    """Raw price-units cloud vs the same cloud after removing store and week means."""
    from .elasticity import demean
    d = panel[panel["upc"] == upc].copy()
    dm = demean(d, ["log_units", "log_price"], ["store_num", "week"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(d["log_price"], d["log_units"], s=3, alpha=0.15)
    b = np.polyfit(d["log_price"], d["log_units"], 1)[0]
    x = np.linspace(d["log_price"].min(), d["log_price"].max(), 10)
    axes[0].plot(x, np.polyval(np.polyfit(d["log_price"], d["log_units"], 1), x), color="C3")
    axes[0].set(title=f"Raw: slope = {b:.2f}", xlabel="log price", ylabel="log units")
    axes[1].scatter(dm["log_price"], dm["log_units"], s=3, alpha=0.15)
    b2 = np.polyfit(dm["log_price"], dm["log_units"], 1)[0]
    x = np.linspace(dm["log_price"].min(), dm["log_price"].max(), 10)
    axes[1].plot(x, np.polyval(np.polyfit(dm["log_price"], dm["log_units"], 1), x), color="C3")
    axes[1].set(title=f"After removing store & week effects: slope = {b2:.2f}",
                xlabel="log price (within)", ylabel="log units (within)")
    fig.suptitle(d["description"].iloc[0])
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------- forecasting
def model_comparison(table, name="fc_model_comparison"):
    t = table.sort_values("WMAPE")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(t.index, t["WMAPE"], color=["C2" if ("XGB" in i or "Light" in i) else "C7" for i in t.index])
    for i, v in enumerate(t["WMAPE"]):
        ax.text(v, i, f" {v:.3f}", va="center")
    ax.invert_yaxis()
    ax.set(title="2011 test period: WMAPE (lower is better)", xlabel="WMAPE")
    return _save(fig, name)


def actual_vs_forecast(df_test, preds: dict, upc=None, name="fc_actual_vs_forecast"):
    d = df_test.copy()
    for k, v in preds.items():
        d[k] = v
    if upc is not None:
        d = d[d["upc"] == upc]
    agg = d.groupby("week")[["units"] + list(preds)].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(agg.index, agg["units"], color="black", lw=2, label="actual")
    for k in preds:
        ax.plot(agg.index, agg[k], lw=1.2, label=k)
    title = "Weekly units, all stores" + (f" - {d['description'].iloc[0]}" if upc is not None else " and products")
    ax.set(title=title + " (2011 test)", ylabel="units")
    ax.legend(fontsize=8)
    return _save(fig, name)


# ---------------------------------------------------------------- elasticity
def elasticity_ladder(ladder, upcs, name="el_ladder"):
    d = ladder[ladder["upc"].isin(upcs) & ladder["spec"].str.startswith(("M1", "M2", "M3", "M4"))]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(upcs) + 1.5))
    specs = sorted(d["spec"].unique())
    for k, s in enumerate(specs):
        x = d[d["spec"] == s]
        y = np.arange(len(upcs)) + (k - 1.5) * 0.18
        pos = {u: i for i, u in enumerate(upcs)}
        yy = [pos[u] + (k - 1.5) * 0.18 for u in x["upc"]]
        ax.errorbar(x["elasticity"], yy, xerr=1.96 * x["se"], fmt="o", ms=4, label=s, capsize=2)
    labels = ladder.drop_duplicates("upc").set_index("upc").loc[upcs, "description"]
    ax.set_yticks(range(len(upcs)), labels)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(-1, color="grey", lw=0.8, ls="--")
    ax.set(title="How the price coefficient changes as controls are added (95% CI)", xlabel="elasticity")
    ax.legend(fontsize=8, loc="lower left")
    return _save(fig, name)


def decision_elasticities(sel, name="el_decision_elasticities"):
    d = sel.dropna(subset=["elasticity"]).sort_values("elasticity")
    fig, ax = plt.subplots(figsize=(9, 0.22 * len(d) + 1.5))
    colors = {c: f"C{i}" for i, c in enumerate(sorted(d["category"].unique()))}
    for i, r in enumerate(d.itertuples()):
        ax.errorbar(r.elasticity, i, xerr=1.96 * r.se, fmt="o" if r.source.startswith("product") else "s",
                    color=colors[r.category], ms=4, capsize=2)
    ax.set_yticks(range(len(d)), d["description"], fontsize=7)
    ax.axvline(-1, color="grey", ls="--", lw=0.8)
    for c, col in colors.items():
        ax.plot([], [], "o", color=col, label=c)
    ax.legend(fontsize=7, loc="lower right")
    ax.set(title="Decision elasticities: promotion-identified, shrunk (product level)",
           xlabel="regular-price elasticity (95% CI)")
    return _save(fig, name)


# ---------------------------------------------------------------- optimization
def objective_curves(r: dict, title="", name=None):
    from .optimization import objective_curve
    c = objective_curve(r["demand_current"], r["current_price"], r["elasticity"], r["unit_cost_assumed"])
    cols = ["demand", "revenue"] + (["profit"] if "profit" in c else [])
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 3.8))
    for ax, col in zip(axes, cols):
        ax.plot(c["price"], c[col], color="C0")
        ax.axvspan(r["lower_bound"], r["upper_bound"], color="C2", alpha=0.12, label="feasible region")
        ax.axvline(r["current_price"], color="grey", ls="--", label="current")
        ax.axvline(r["recommended_price"], color="C3", label="recommended")
        ax.set(title=col.capitalize() + (" (assumed cost)" if col == "profit" else ""), xlabel="price")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{title}  (E = {r['elasticity']:.2f})")
    fig.tight_layout()
    return _save(fig, name)


def backtest_scenarios(summary, name="bt_scenarios"):
    d = summary.set_index("scenario")[["revenue_change_pct", "profit_change_pct", "units_change_pct"]] * 100
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(d) + 1.5))
    d.plot.barh(ax=ax)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set(title="Offline counterfactual simulation, 2011 (SIMULATED % change vs historical pricing)",
           xlabel="% change (model-based estimate, not observed)")
    return _save(fig, name)


def price_response_validation(val, name="bt_price_response_validation"):
    d = val[val["model"] != "No price effect (E = 0)"].set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * len(d) + 1.8), sharey=True)
    axes[0].barh(d.index, d["MAE_skill_within_week"] * 100, color=["C3" if "ENGINE" in i else "C0" for i in d.index])
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set(title="Prediction skill vs 'no price effect' (%)", xlabel="% lower MAE (higher is better)")
    axes[1].barh(d.index, d["calibration_slope"], color=["C3" if "ENGINE" in i else "C0" for i in d.index])
    axes[1].axvline(1, color="black", lw=0.8, ls="--")
    axes[1].set(title="Calibration slope (1 = correct magnitude)", xlabel="observed lift / predicted lift")
    axes[0].invert_yaxis()
    fig.suptitle("Held-out validation: 2009-10 elasticities predicting real 2011 price-cut lifts")
    fig.tight_layout()
    return _save(fig, name)


def dynamic_paths(schedule, panel_test, name="bt_dynamic_price_paths"):
    s = schedule.merge(panel_test.groupby("upc")[["category"]].first(), on="upc")
    fig, ax = plt.subplots(figsize=(12, 4))
    for (cat, seg), g in s.groupby(["category", "store_segment"]):
        m = g.groupby("week")["multiplier"].median()
        ax.plot(m.index, (m - 1) * 100, lw=1.3, label=f"{cat.title()} - {seg.title()}")
    ax.set(title="Dynamic engine in the 2011 walk-forward backtest: median price change vs reference (%)",
           ylabel="% vs reference regular price")
    ax.legend(fontsize=7, ncol=3)
    return _save(fig, name)
