"""Price elasticity estimation on observational scanner data.

Model family (log-log / constant elasticity):
    log(Q_ist) = beta * log(P_ist) + controls + e_ist
beta is the elasticity: a 1% higher price is associated with ~beta% change in units,
*under the model specification*. This is an association conditional on the
controls, not a guaranteed causal effect (see notebook 03 and PROJECT_DECISIONS.md).

Specification ladder (estimated per product):
    M1 naive          : log Q ~ log P
    M2 + promo        : log Q ~ log P + feature + display
    M3 + store FE + seasonality (month, year, holiday dummies)
    M4 + store FE + week FE ("two-way fixed effects")      <- main controlled model
    M5 decomposition  : M4 with log P split into
                        log(base price)          -> regular-price elasticity
                        log(price / base price)  -> temporary-discount elasticity

Fixed effects are removed with the within transformation (alternating projections):
by the Frisch-Waugh-Lovell theorem, OLS on the demeaned variables gives the same
slope as OLS with the full set of dummies, at a fraction of the memory.
Standard errors are clustered by store (errors of the same store are correlated
over time).

Note: only weeks with positive recorded sales enter log Q. Weeks without a
record (zero demand or stockout - indistinguishable) are excluded. This is a
selection issue and is listed as a limitation.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

# M3 can contain redundant year dummies for products that do not span all years.
# statsmodels then uses the pseudo-inverse; the price coefficient is still identified.
warnings.filterwarnings("ignore", category=SingularMatrixWarning)

from . import config


# --------------------------------------------------------------------------
# Price variation analysis
# --------------------------------------------------------------------------
def price_variation_table(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    key = ["store_num", "upc"]
    for c in ("log_price", "log_base_price", "log_discount_ratio"):
        d[c + "_w"] = d[c] - d.groupby(key)[c].transform("mean")
    d["base_change"] = (d.groupby(key)["base_price"].diff().abs() > 0.005).astype(int)
    t = d.groupby(["upc", "description", "category"]).agg(
        n_obs=("units", "size"),
        n_stores=("store_num", "nunique"),
        distinct_prices=("price", "nunique"),
        mean_price=("price", "mean"),
        mean_units=("units", "mean"),
        sd_log_price_within=("log_price_w", "std"),
        sd_log_base_within=("log_base_price_w", "std"),
        sd_log_discount_within=("log_discount_ratio_w", "std"),
        share_weeks_discounted=("on_discount", "mean"),
        share_weeks_feature=("feature", "mean"),
        share_weeks_display=("display", "mean"),
        base_price_changes=("base_change", "sum"),
    ).reset_index()
    # share of within-store price variation that comes from temporary discounts
    # (log price = log base price + log discount ratio)
    vd, vb = t["sd_log_discount_within"] ** 2, t["sd_log_base_within"] ** 2
    t["share_var_from_discounts"] = vd / (vd + vb)
    t["eligible"] = (
        (t["n_obs"] >= config.MIN_OBS)
        & (t["distinct_prices"] >= config.MIN_DISTINCT_PRICES)
        & (t["sd_log_price_within"] >= config.MIN_WITHIN_LOG_PRICE_SD)
    )
    t["eligible_regular_price"] = t["eligible"] & (t["sd_log_base_within"] >= config.MIN_WITHIN_LOG_PRICE_SD)
    return t.sort_values("sd_log_price_within", ascending=False)


# --------------------------------------------------------------------------
# Fixed-effects helpers
# --------------------------------------------------------------------------
def demean(df: pd.DataFrame, cols, fe_cols, tol=1e-10, max_iter=500) -> pd.DataFrame:
    """Remove one or more sets of fixed effects by alternating projections."""
    out = df[cols].astype(float).copy()
    if len(fe_cols) == 1:
        return out - out.groupby(df[fe_cols[0]]).transform("mean")
    for _ in range(max_iter):
        prev = out.copy()
        for fe in fe_cols:
            out = out - out.groupby(df[fe]).transform("mean")
        if np.max(np.abs((out - prev).values)) < tol:
            break
    return out


def _ols(y, X, groups=None, add_const=False):
    if add_const:
        X = sm.add_constant(X, has_constant="add")
    m = sm.OLS(y, X)
    if groups is not None:
        return m.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return m.fit(cov_type="HC1")


def _row(res, var, spec, **extra):
    b, se = res.params[var], res.bse[var]
    return {"spec": spec, "elasticity": b, "se": se, "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se,
            "p_value": res.pvalues[var], "r2": res.rsquared, **extra}


# --------------------------------------------------------------------------
# Per-product specification ladder
# --------------------------------------------------------------------------
SEASON_DUMMY_COLS = None


def _season_dummies(d):
    s = pd.get_dummies(d["week"].dt.month.astype(str).radd("m"), drop_first=True, dtype=float)
    s = s.join(pd.get_dummies(d["week"].dt.year.astype(str).radd("y"), drop_first=True, dtype=float))
    return s


def estimate_product(d: pd.DataFrame, holiday: pd.Series | None = None) -> list[dict]:
    """Run M1-M5 for one product's store-week rows. Returns list of result dicts."""
    d = d.copy()
    out = []
    g = d["store_num"].values
    n = len(d)

    # M1 naive
    r1 = _ols(d["log_units"], d[["log_price"]], add_const=True)
    out.append(_row(r1, "log_price", "M1 naive", n=n))

    # M2 promo controls
    r2 = _ols(d["log_units"], d[["log_price", "feature", "display"]], groups=g, add_const=True)
    out.append(_row(r2, "log_price", "M2 + promo controls", n=n))

    # M3 store FE + seasonality
    seas = _season_dummies(d)
    if holiday is not None:
        seas["holiday"] = holiday.loc[d.index].values
    X3 = pd.concat([d[["log_price", "feature", "display"]], seas.set_index(d.index)], axis=1)
    cols3 = ["log_units"] + list(X3.columns)
    dm3 = demean(pd.concat([d[["log_units", "store_num"]], X3], axis=1), cols3, ["store_num"])
    dm3 = dm3.loc[:, dm3.std() > 1e-12]
    r3 = _ols(dm3["log_units"], dm3.drop(columns="log_units"), groups=g)
    out.append(_row(r3, "log_price", "M3 + store FE + seasonality", n=n))

    # M4 two-way FE (store + week)
    cols4 = ["log_units", "log_price", "feature", "display"]
    dm4 = demean(d, cols4, ["store_num", "week"])
    dm4 = dm4.loc[:, dm4.std() > 1e-12]
    r4 = _ols(dm4["log_units"], dm4.drop(columns="log_units"), groups=g)
    promo = {"feature_coef": r4.params.get("feature", np.nan), "display_coef": r4.params.get("display", np.nan)}
    out.append(_row(r4, "log_price", "M4 + store FE + week FE", n=n, **promo))

    # M5 decomposition under two-way FE
    cols5 = ["log_units", "log_base_price", "log_discount_ratio", "feature", "display"]
    dm5 = demean(d, cols5, ["store_num", "week"])
    dm5 = dm5.loc[:, dm5.std() > 1e-12]
    r5 = _ols(dm5["log_units"], dm5.drop(columns="log_units"), groups=g)
    if "log_base_price" in r5.params:
        out.append(_row(r5, "log_base_price", "M5 regular-price elasticity", n=n))
    if "log_discount_ratio" in r5.params:
        out.append(_row(r5, "log_discount_ratio", "M5 temporary-discount elasticity", n=n))
    return out


def estimate_all_products(panel: pd.DataFrame, variation: pd.DataFrame, holiday=None,
                          exclude_outliers=False) -> pd.DataFrame:
    """Specification ladder for every product that meets the minimum-data rules."""
    rows = []
    eligible = variation.loc[variation["eligible"], "upc"]
    for upc in eligible:
        d = panel[panel["upc"] == upc]
        if exclude_outliers:
            d = d[d["is_outlier"] == 0]
        for r in estimate_product(d, holiday):
            rows.append({"upc": upc, **r})
    res = pd.DataFrame(rows)
    info = variation[["upc", "description", "category", "mean_price", "mean_units"]]
    return info.merge(res, on="upc", how="right")


def estimate_category_pooled(panel: pd.DataFrame, upcs=None) -> pd.DataFrame:
    """Pooled elasticity per category: one common slope, with store x product
    FE and product x week FE (i.e. each product keeps its own seasonality).
    Used as a fallback for products whose own estimate is unreliable."""
    rows = []
    d0 = panel if upcs is None else panel[panel["upc"].isin(upcs)]
    for cat, d in d0.groupby("category"):
        d = d.copy()
        d["sp"] = d["store_num"].astype(str) + "_" + d["upc"].astype(str)
        d["pw"] = d["upc"].astype(str) + "_" + d["week"].astype(str)
        for spec, cols, var in (
            ("pooled shelf-price", ["log_units", "log_price", "feature", "display"], "log_price"),
            ("pooled regular-price", ["log_units", "log_base_price", "log_discount_ratio", "feature", "display"],
             "log_base_price"),
        ):
            dm = demean(d, cols, ["sp", "pw"])
            r = _ols(dm["log_units"], dm.drop(columns="log_units"), groups=d["store_num"].values)
            rows.append({"category": cat, **_row(r, var, spec, n=len(d), n_products=d["upc"].nunique())})
    return pd.DataFrame(rows)


def holiday_series(panel):
    """Holiday-week dummy aligned to panel rows (for M3)."""
    from .feature_engineering import _calendar_features
    cal = _calendar_features(panel["week"])
    return cal["holiday_week"]


# ==========================================================================
# PROMOTION-BASED IDENTIFICATION (decision elasticities)
# ==========================================================================
# Problem with plain OLS: the store's price is a choice that can react to local
# demand (endogeneity), and promotions bundle a price cut with ads/displays
# (confounding). Promotion-driven price variation helps in two ways:
#
#  M6  "TPR-only" sample: keep only weeks with no promotion or a temporary price
#      reduction WITHOUT feature ad or display. The price variation that remains
#      is a pure price cut, so the advertising/display confound is removed by design.
#
#  M7  Promotion-calendar instrumental variable (2SLS): in this data ~74% of the
#      variance of discount depth is chain-wide (same product, same week, all
#      stores), i.e. set centrally - typically weeks in advance and often funded
#      by manufacturer trade deals. We instrument a store's log price with the
#      AVERAGE discount depth of the same product in the OTHER stores that week
#      (leave-one-out). This uses only the centrally planned promotion calendar,
#      not the store's own (possibly demand-reactive) price decision.
#      Relevance : the chain calendar moves the store price (first-stage F reported).
#      Exclusion : the calendar affects the store's demand only through its price,
#                  given the store's own feature/display, store effects and
#                  seasonality (month, year, holiday). THIS IS AN ASSUMPTION: it fails
#                  if the chain times promotions to demand peaks not captured by the
#                  seasonal controls, or if national advertising coincides with deals.
#      Week fixed effects cannot be used here, because the instrument varies
#      mainly by week. Seasonality dummies are used instead.
#
# Both reduce, but do not eliminate, confounding. They are reported as such.


def add_promo_instrument(panel: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out mean log(price/base price) of the same product in the other stores that week."""
    d = panel.copy()
    g = d.groupby(["upc", "week"])["log_discount_ratio"]
    s, n = g.transform("sum"), g.transform("count")
    d["n_stores_week"] = n
    d["z_promo_calendar"] = np.where(n > 1, (s - d["log_discount_ratio"]) / (n - 1).clip(lower=1), np.nan)
    d["discount_lag_1"] = d.sort_values("week").groupby(["store_num", "upc"])["discount_pct"].shift(1).fillna(0)
    return d


def _seasonal_exog(d):
    s = _season_dummies(d)
    s.index = d.index
    s["holiday"] = holiday_series(d).values
    return s


def estimate_tpr_only(d: pd.DataFrame) -> dict:
    """M6: drop feature/display weeks; store + week fixed effects."""
    x = d[(d["feature"] == 0) & (d["display"] == 0)]
    dm = demean(x, ["log_units", "log_price"], ["store_num", "week"])
    r = _ols(dm["log_units"], dm[["log_price"]], groups=x["store_num"].values)
    return _row(r, "log_price", "M6 TPR-only (no ad/display weeks)", n=len(x))


def _iv_fit(d, endog_cols, instr_cols, exog_df):
    from linearmodels.iv import IV2SLS
    cols = ["log_units"] + endog_cols + instr_cols + list(exog_df.columns)
    base = pd.concat([d[["log_units", "store_num"] + endog_cols + instr_cols], exog_df], axis=1)
    dm = demean(base, cols, ["store_num"])
    keep_exog = [c for c in exog_df.columns if dm[c].std() > 1e-10]
    mod = IV2SLS(dm["log_units"], dm[keep_exog] if keep_exog else None, dm[endog_cols], dm[instr_cols])
    return mod.fit(cov_type="clustered", clusters=d["store_num"])


def estimate_iv_product(d: pd.DataFrame) -> dict:
    """M7: 2SLS with the leave-one-out promotion-calendar instrument (all stores pooled)."""
    d = d.dropna(subset=["z_promo_calendar"])
    exog = pd.concat([d[["feature", "display", "discount_lag_1"]], _seasonal_exog(d)], axis=1)
    r = _iv_fit(d, ["log_price"], ["z_promo_calendar"], exog)
    fs = r.first_stage.diagnostics
    b, se = float(r.params["log_price"]), float(r.std_errors["log_price"])
    return {"spec": "M7 promotion-calendar IV", "elasticity": b, "se": se, "ci_low": b - 1.96 * se,
            "ci_high": b + 1.96 * se, "p_value": float(r.pvalues["log_price"]), "r2": np.nan, "n": len(d),
            "first_stage_F": float(fs.loc["log_price", "f.stat"])}


def estimate_iv_segments(d: pd.DataFrame) -> list[dict]:
    """M7 with store-segment interactions: one elasticity per segment (MAINSTREAM base)."""
    d = d.dropna(subset=["z_promo_calendar"]).copy()
    segs = [s for s in ("VALUE", "UPSCALE") if (d["store_segment"] == s).sum() > 200]
    endog, instr = ["log_price"], ["z_promo_calendar"]
    for s in segs:
        d[f"lp_{s}"] = d["log_price"] * (d["store_segment"] == s)
        d[f"z_{s}"] = d["z_promo_calendar"] * (d["store_segment"] == s)
        endog.append(f"lp_{s}"); instr.append(f"z_{s}")
    exog = pd.concat([d[["feature", "display", "discount_lag_1"]], _seasonal_exog(d)], axis=1)
    r = _iv_fit(d, endog, instr, exog)
    out = [{"segment": "MAINSTREAM", "deviation": 0.0, "deviation_se": 0.0,
            "base": float(r.params["log_price"]), "base_se": float(r.std_errors["log_price"])}]
    for s in segs:
        out.append({"segment": s, "deviation": float(r.params[f"lp_{s}"]),
                    "deviation_se": float(r.std_errors[f"lp_{s}"]),
                    "base": float(r.params["log_price"]), "base_se": float(r.std_errors["log_price"])})
    return out


# --------------------------------------------------------------------------
# Empirical-Bayes shrinkage (partial pooling)
# --------------------------------------------------------------------------
def shrink(b, se, prior_mean=None):
    """Shrink noisy estimates toward a common mean (DerSimonian-Laird random effects).

    shrunk_i = mu + w_i (b_i - mu),  w_i = tau^2 / (tau^2 + se_i^2)
    tau^2 = between-product variance of the true elasticities (method of moments).
    Precise estimates barely move; noisy ones move toward the group mean.
    If prior_mean is given (e.g. 0 for segment deviations) it is used as mu.
    """
    b, se = np.asarray(b, float), np.asarray(se, float)
    w = 1 / se ** 2
    k = len(b)
    if k < 2:
        return b, se, 0.0, (b[0] if prior_mean is None else prior_mean)
    bw = np.sum(w * b) / np.sum(w)
    Q = np.sum(w * (b - bw) ** 2)
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    if prior_mean is None:
        ws = 1 / (se ** 2 + tau2)
        mu = np.sum(ws * b) / np.sum(ws)
    else:
        mu = prior_mean
    lam = tau2 / (tau2 + se ** 2) if tau2 > 0 else np.zeros_like(b)
    shrunk = mu + lam * (b - mu)
    post_se = np.sqrt(lam * se ** 2) if tau2 > 0 else np.full_like(b, np.nan)
    return shrunk, post_se, tau2, mu


def estimate_promo_identified(panel: pd.DataFrame, variation: pd.DataFrame, with_iv: bool = True):
    """Promotion-based estimates for every eligible product.

    M6 (TPR-only sample, shelf price), M7 (promotion-calendar IV - a DIAGNOSTIC that
    turned out to be invalid, see notebook 03) and M8 (the decision model) +
    M8 store-segment interactions. Returns (product_table, segment_table)."""
    d0 = add_promo_instrument(panel)
    prod_rows, seg_rows = [], []
    for upc in variation.loc[variation["eligible"], "upc"]:
        d = d0[d0["upc"] == upc]
        prod_rows.append({"upc": upc, **estimate_tpr_only(d)})
        if with_iv:
            prod_rows.append({"upc": upc, **estimate_iv_product(d)})
        prod_rows.append({"upc": upc, **estimate_promo_identified_product(d)})
        for s in estimate_promo_identified_segments(d):
            seg_rows.append({"upc": upc, **s})
    info = variation[["upc", "description", "category", "mean_price", "mean_units"]]
    return info.merge(pd.DataFrame(prod_rows), on="upc", how="right"), \
        info[["upc", "description", "category"]].merge(pd.DataFrame(seg_rows), on="upc", how="right")


DECISION_SPEC = "M8 promotion-identified (TPR events, store+week FE)"


def decision_elasticities(prod: pd.DataFrame, seg: pd.DataFrame):
    """Product-level and product x store-segment elasticities used by the pricing engine.

    1. Product level: M8 promotion-identified estimate, shrunk toward its category mean
       (every product gets a product-level estimate; noisy ones borrow strength from
       the category instead of being replaced by it).
    2. Segment level: segment deviations from the product estimate, shrunk toward 0
       across products of the same category, then added to the shrunk product estimate.
    3. A shrunk product elasticity that is still >= -0.1 is not usable for pricing
       (implausible sign / no measurable response) -> the engine keeps the current price.
    4. Segment elasticities are capped at -0.05 so that a segment adjustment never
       flips the sign of a usable product's response.
    """
    m8 = prod[prod["spec"] == DECISION_SPEC].copy()
    out = []
    for cat, g in m8.groupby("category"):
        s, ps, tau2, mu = shrink(g["elasticity"], g["se"])
        out.append(g.assign(elasticity_raw=g["elasticity"], se_raw=g["se"], elasticity=s, se=ps,
                            category_mean=mu, tau=np.sqrt(tau2)))
    prod_dec = pd.concat(out)[["upc", "description", "category", "elasticity_raw", "se_raw", "n_tpr_weeks",
                               "elasticity", "se", "category_mean", "tau"]]
    prod_dec["usable"] = prod_dec["elasticity"] < -0.1
    prod_dec["source"] = np.where(prod_dec["usable"], "product (promotion-identified, shrunk)",
                                  "none - keep current price")

    sd = seg[seg["segment"] != "MAINSTREAM"].copy()
    parts = []
    for (cat, sg), g in sd.groupby(["category", "segment"]):
        s, ps, tau2, _ = shrink(g["deviation"], g["deviation_se"], prior_mean=0.0)
        parts.append(g.assign(deviation_shrunk=s))
    sd = pd.concat(parts) if parts else sd.assign(deviation_shrunk=[])
    base = seg[seg["segment"] == "MAINSTREAM"].assign(deviation_shrunk=0.0)
    segs = pd.concat([base, sd])[["upc", "segment", "deviation", "deviation_se", "deviation_shrunk"]]
    # complete grid: a segment whose deviation could not be estimated gets deviation 0,
    # i.e. it uses the product estimate (the limit of shrinkage with no information)
    grid = pd.MultiIndex.from_product([prod_dec["upc"].unique(), ["MAINSTREAM", "VALUE", "UPSCALE"]],
                                      names=["upc", "segment"]).to_frame(index=False)
    segs = grid.merge(segs, on=["upc", "segment"], how="left")
    segs["estimated"] = segs["deviation"].notna()
    segs[["deviation", "deviation_se", "deviation_shrunk"]] = segs[["deviation", "deviation_se", "deviation_shrunk"]].fillna(0.0)
    segs = segs.merge(prod_dec[["upc", "description", "category", "elasticity", "usable"]], on="upc")
    # a segment adjustment may refine, but never flip, a usable product's price response
    segs["segment_elasticity"] = np.minimum(segs["elasticity"] + segs["deviation_shrunk"], -0.05)
    return prod_dec, segs


# --------------------------------------------------------------------------
# M8: promotion-identified elasticity (DECISION MODEL)
# --------------------------------------------------------------------------
def _tpr_sample(d):
    """No-promotion weeks + pure temporary price reductions (no feature ad, no display)."""
    return d[(d["feature"] == 0) & (d["display"] == 0)]


def estimate_promo_identified_product(d: pd.DataFrame) -> dict:
    """Elasticity identified from temporary-price-reduction (TPR) events.

    log Q = b_tpr * log(P / base P) + b_base * log(base P) + g * discount_last_week
            + store FE + week FE + e,    sample = no-promo weeks and TPR-only weeks

    Confounding is addressed by design:
      * feature-ad and display weeks are excluded, so the price cut is not bundled with
        advertising or extra shelf space;
      * store FE remove permanent store differences (size, clientele);
      * week FE remove every chain-wide shock of that week, INCLUDING the chain's
        promotion timing (promotions scheduled into demand peaks). What identifies
        b_tpr is stores running different temporary discounts in the same week;
      * last week's discount controls for pantry loading (post-promotion dip).
    b_tpr is the response to a temporary price change - the kind of week-to-week
    move the dynamic pricing engine makes.
    """
    x = _tpr_sample(d)
    cols = ["log_units", "log_discount_ratio", "log_base_price", "discount_lag_1"]
    dm = demean(x, cols, ["store_num", "week"])
    dm = dm.loc[:, dm.std() > 1e-12]
    r = _ols(dm["log_units"], dm.drop(columns="log_units"), groups=x["store_num"].values)
    out = _row(r, "log_discount_ratio", "M8 promotion-identified (TPR events, store+week FE)", n=len(x))
    out["n_tpr_weeks"] = int((x["log_discount_ratio"] < -0.005).sum())
    out["sd_within_discount"] = float(dm["log_discount_ratio"].std())
    return out


def estimate_promo_identified_segments(d: pd.DataFrame) -> list[dict]:
    """M8 with store-segment interactions on the TPR term (MAINSTREAM = base)."""
    x = _tpr_sample(d).copy()
    segs = [s for s in ("VALUE", "UPSCALE")
            if ((x["store_segment"] == s) & (x["log_discount_ratio"] < -0.005)).sum() > 100]
    inter = []
    for s in segs:
        x[f"ld_{s}"] = x["log_discount_ratio"] * (x["store_segment"] == s)
        inter.append(f"ld_{s}")
    cols = ["log_units", "log_discount_ratio", "log_base_price", "discount_lag_1"] + inter
    dm = demean(x, cols, ["store_num", "week"])
    dm = dm.loc[:, dm.std() > 1e-12]
    r = _ols(dm["log_units"], dm.drop(columns="log_units"), groups=x["store_num"].values)
    out = [{"segment": "MAINSTREAM", "deviation": 0.0, "deviation_se": 0.0}]
    for s in segs:
        if f"ld_{s}" in r.params:
            out.append({"segment": s, "deviation": float(r.params[f"ld_{s}"]),
                        "deviation_se": float(r.bse[f"ld_{s}"])})
    return out
