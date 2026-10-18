"""Time-aware feature engineering for weekly store x product demand.

Leakage rules followed here:
  * All demand-history features (lags, rolling stats) are computed on a complete
    weekly calendar per store-product and SHIFTED by >= 1 week, so the value for
    week t only uses weeks <= t-1.
  * Same-week outcome variables (VISITS, HHS, SPEND) are never used as features:
    they are only known after the week is over and are mechanically tied to UNITS.
  * Price and promotion flags of week t ARE used. They are decision variables the
    retailer plans in advance (price list, circular, display plan), so they are
    known at forecast time. This is exactly what lets us ask "what if price = P?".

Because the data is weekly, the lag set is weekly: 1, 2, 3, 4, 8, 13 and 52 weeks
(52 = same week last year = yearly seasonality).
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import config

LAGS = [1, 2, 3, 4, 8, 13, 52]
ROLL_WINDOWS = [4, 8, 13]

CATEGORICAL = ["store_num", "upc", "category", "sub_category", "manufacturer",
               "store_segment", "state"]
PRICE_PROMO = ["price", "base_price", "discount_pct", "log_price", "feature", "display",
               "tpr_only", "on_discount", "price_vs_recent_base", "base_vs_recent_base"]
CALENDAR = ["week_of_year", "month", "quarter", "year", "holiday_week", "thanksgiving_week",
            "christmas_week", "super_bowl_week"]
STORE_NUMERIC = ["sales_area_size_num", "avg_weekly_baskets"]


def _calendar_features(weeks: pd.Series) -> pd.DataFrame:
    w = pd.to_datetime(weeks)
    out = pd.DataFrame(index=weeks.index)
    out["week_of_year"] = w.dt.isocalendar().week.astype(int)
    out["month"] = w.dt.month
    out["quarter"] = w.dt.quarter
    out["year"] = w.dt.year
    # A week "contains" a date if the date is in (week_end - 6 days, week_end]
    hol = USFederalHolidayCalendar().holidays(start="2008-12-01", end="2012-02-01")

    def contains(dates, name=None):
        uniq = pd.Series(w.unique())
        flags = {d: int(any((h > d - pd.Timedelta(days=7)) and (h <= d) for h in dates)) for d in uniq}
        return w.map(flags).astype(int)

    out["holiday_week"] = contains(hol)
    thanks = [h for h in hol if h.month == 11 and h.day >= 22]
    xmas = [h for h in hol if h.month == 12 and h.day == 25] + list(pd.to_datetime(["2009-12-25", "2010-12-24", "2011-12-26"]))
    out["thanksgiving_week"] = contains(thanks)
    out["christmas_week"] = contains(xmas)
    out["super_bowl_week"] = contains(list(config.SUPER_BOWL_DATES))
    return out


def build_grid(panel: pd.DataFrame, future_week: pd.Timestamp | None = None,
               active_weeks: int = 4) -> pd.DataFrame:
    """Complete weekly calendar for each store-product between its first and last sale.

    Weeks with no record get NaN units (NOT zero) - see the stockout discussion.
    If `future_week` is given, the calendar of every store-product that sold in any
    of the last `active_weeks` weeks is extended to the last observed week (gap
    weeks stay NaN, so lag_1 really means "last week") and one extra row is
    appended for `future_week` (used to forecast the next, unobserved week).
    """
    all_weeks = pd.Series(sorted(panel["week"].unique()))
    last = all_weeks.iloc[-1]
    spans = panel.groupby(["store_num", "upc"])["week"].agg(["min", "max"]).reset_index()
    if future_week is not None:
        recent = spans["max"] >= all_weeks.iloc[-active_weeks]
        spans.loc[recent, "max"] = last
    rows = []
    for r in spans.itertuples(index=False):
        wk = all_weeks[(all_weeks >= r.min) & (all_weeks <= r.max)]
        rows.append(pd.DataFrame({"store_num": r.store_num, "upc": r.upc, "week": wk.values}))
    grid = pd.concat(rows, ignore_index=True)
    grid = grid.merge(panel, on=["store_num", "upc", "week"], how="left")
    grid["observed"] = grid["units"].notna().astype(int)

    if future_week is not None:
        active = spans.loc[spans["max"] == last, ["store_num", "upc"]]
        fut = active.assign(week=future_week, observed=0)
        grid = pd.concat([grid, fut], ignore_index=True)

    # static attributes must exist on every grid row (including gaps / future rows)
    static_store = panel.groupby("store_num")[["store_segment", "state", "sales_area_size_num",
                                               "avg_weekly_baskets", "store_name"]].first()
    static_prod = panel.groupby("upc")[["category", "sub_category", "manufacturer",
                                        "description", "product_size"]].first()
    grid = grid.drop(columns=[c for c in static_store.columns if c in grid], errors="ignore")
    grid = grid.drop(columns=[c for c in static_prod.columns if c in grid], errors="ignore")
    grid = grid.join(static_store, on="store_num").join(static_prod, on="upc")
    return grid.sort_values(["store_num", "upc", "week"]).reset_index(drop=True)


def add_features(grid: pd.DataFrame) -> pd.DataFrame:
    df = grid.copy()
    g = df.groupby(["store_num", "upc"], sort=False)

    # ---- demand history (strictly past) ----
    for k in LAGS:
        df[f"units_lag_{k}"] = g["units"].shift(k)
    past = g["units"].shift(1)  # value known at the start of week t
    pg = past.groupby([df["store_num"], df["upc"]], sort=False)
    for w in ROLL_WINDOWS:
        df[f"units_rollmean_{w}"] = pg.transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["units_rollstd_8"] = pg.transform(lambda s: s.rolling(8, min_periods=2).std())
    df["units_expanding_mean"] = pg.transform(lambda s: s.expanding().mean())
    # how many of the last 13 weeks had any recorded sale (availability / ranging proxy)
    obs_past = g["observed"].shift(1)
    df["weeks_observed_last_13"] = obs_past.groupby([df["store_num"], df["upc"]], sort=False) \
        .transform(lambda s: s.rolling(13, min_periods=1).sum())

    # ---- promotion history (pantry-loading / post-promotion dip) ----
    df["discount_lag_1"] = g["discount_pct"].shift(1)
    df["feature_lag_1"] = g["feature"].shift(1)
    df["display_lag_1"] = g["display"].shift(1)

    # ---- price relative to the recent REGULAR price (past only) ----
    base_past = g["base_price"].shift(1)
    recent_base = base_past.groupby([df["store_num"], df["upc"]], sort=False) \
        .transform(lambda s: s.rolling(13, min_periods=1).median())
    df["recent_base_price"] = recent_base  # helper, not a model feature
    df["price_vs_recent_base"] = df["price"] / recent_base
    df["base_vs_recent_base"] = df["base_price"] / recent_base

    # ---- calendar ----
    cal = _calendar_features(df["week"])
    df = pd.concat([df, cal], axis=1)

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


CATEGORY_FILE = config.MODELS_DIR / "category_levels.json"


def save_category_levels(df: pd.DataFrame) -> None:
    """Persist the category levels seen at training time. XGBoost/LightGBM map a
    categorical value to a code by its position in the level list, so scoring
    data MUST use exactly the same levels in the same order."""
    import json
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    levels = {c: [v.item() if hasattr(v, "item") else v for v in df[c].astype("category").cat.categories]
              for c in CATEGORICAL}
    CATEGORY_FILE.write_text(json.dumps(levels))


def apply_category_levels(df: pd.DataFrame) -> pd.DataFrame:
    import json
    levels = json.loads(CATEGORY_FILE.read_text())
    df = df.copy()
    for c in CATEGORICAL:
        vals = df[c].astype(object) if not isinstance(df[c].dtype, pd.CategoricalDtype) else df[c].astype(object)
        df[c] = pd.Categorical(vals, categories=levels[c])
    return df


def feature_columns():
    hist = [f"units_lag_{k}" for k in LAGS] + [f"units_rollmean_{w}" for w in ROLL_WINDOWS] + \
           ["units_rollstd_8", "units_expanding_mean", "weeks_observed_last_13",
            "discount_lag_1", "feature_lag_1", "display_lag_1"]
    return CATEGORICAL + STORE_NUMERIC + PRICE_PROMO + CALENDAR + hist


def build_feature_table(panel: pd.DataFrame, future_week=None) -> pd.DataFrame:
    """Grid -> features. Training rows are those with observed units."""
    feats = add_features(build_grid(panel, future_week=future_week))
    return feats


def set_price_scenario(rows: pd.DataFrame, price, base_price=None, feature=0, display=0) -> pd.DataFrame:
    """Overwrite the price/promotion inputs of feature rows consistently.

    Used for (a) forecasting next week under a planned price and (b) the
    'model price response' sanity check. All price-derived columns are recomputed
    from the same numbers so the inputs never contradict each other.
    """
    r = rows.copy()
    recent_base = r["recent_base_price"]
    r["price"] = price
    r["base_price"] = np.maximum(price, base_price if base_price is not None else price)
    r["discount_pct"] = (1 - r["price"] / r["base_price"]).clip(lower=0)
    r["on_discount"] = (r["discount_pct"] > 0.005).astype(int)
    r["tpr_only"] = ((r["on_discount"] == 1) & (feature == 0) & (display == 0)).astype(int)
    r["feature"] = feature
    r["display"] = display
    r["log_price"] = np.log(r["price"])
    r["price_vs_recent_base"] = r["price"] / recent_base
    r["base_vs_recent_base"] = r["base_price"] / recent_base
    return r
