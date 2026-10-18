"""Cleaning pipeline.

Design principle: never silently delete data. Every step appends an entry to a
cleaning log (step, rows affected, action, reason) that is saved to
reports/results/cleaning_log.csv and discussed in notebook 01.
"""
import numpy as np
import pandas as pd


def _log(log, step, n, action, reason):
    log.append({"step": step, "rows_affected": int(n), "action": action, "reason": reason})


def clean_transactions(tx: pd.DataFrame, stores: pd.DataFrame, products: pd.DataFrame):
    """Return (clean panel, cleaning log DataFrame)."""
    log = []
    df = tx.copy()
    df = df.rename(columns={"week_end_date": "week"})
    df["week"] = pd.to_datetime(df["week"])
    n0 = len(df)
    _log(log, "raw rows", n0, "none", "starting point")

    # 1. Duplicates on the panel key (week, store, product)
    key = ["week", "store_num", "upc"]
    dups = df.duplicated(key).sum()
    df = df.drop_duplicates(key)
    _log(log, "duplicate (week, store, upc)", dups, "drop",
         "the panel key must be unique; none were found")

    # 2. Missing PRICE. SPEND = PRICE x UNITS holds exactly in this data
    #    (verified in notebook 01), so price is recoverable when units > 0.
    miss_price = df["price"].isna() | (df["price"] <= 0)
    recover = miss_price & (df["units"] > 0) & (df["spend"] > 0)
    df.loc[recover, "price"] = df.loc[recover, "spend"] / df.loc[recover, "units"]
    _log(log, "missing/zero price, units>0", recover.sum(), "impute price = spend/units",
         "accounting identity SPEND = PRICE*UNITS holds exactly in the data")

    # 3. Rows with zero units (all have missing price and zero spend). With VISITS>0
    #    these are most likely voids/returns; they carry no price information.
    zero_units = df["units"] <= 0
    _log(log, "units == 0 (with visits > 0)", zero_units.sum(), "drop",
         "no price recorded and no units; likely voided transactions, not 'zero demand'")
    df = df[~zero_units]

    still_missing = df["price"].isna() | (df["price"] <= 0)
    _log(log, "price still missing", still_missing.sum(), "drop", "price not recoverable")
    df = df[~still_missing]

    # 4. Missing BASE_PRICE: regular price is 'sticky', so carry it forward/backward
    #    within the same store-product; fall back to the shelf price.
    df = df.sort_values(key).reset_index(drop=True)
    miss_base = df["base_price"].isna()
    df["base_price"] = df.groupby(["store_num", "upc"])["base_price"].transform(lambda s: s.ffill().bfill())
    df["base_price"] = df["base_price"].fillna(df["price"])
    _log(log, "missing base_price", miss_base.sum(), "ffill/bfill within store-product",
         "regular price changes rarely; neighbouring weeks are the best estimate")

    # 5. Shelf price above base price (inconsistent by definition of base price).
    above = df["price"] > df["base_price"] + 0.01
    df["base_price_adjusted"] = above.astype(int)
    df.loc[above, "base_price"] = df.loc[above, "price"]
    _log(log, "price > base_price", above.sum(), "set base_price = price (flagged)",
         "base price is the regular (non-discounted) price, so it cannot be below the shelf price;"
         " most likely a lag in the base-price field")

    # 6. Outliers: flag, do not delete. Extreme units relative to the store-product's
    #    own history. Real demand spikes (e.g. Super Bowl) are legitimate.
    g = df.groupby(["store_num", "upc"])["units"]
    med = g.transform("median")
    iqr = g.transform(lambda s: s.quantile(0.75) - s.quantile(0.25)).clip(lower=1)
    df["is_outlier"] = (df["units"] > med + 10 * iqr).astype(int)
    _log(log, "extreme units (> median + 10*IQR within store-product)", df["is_outlier"].sum(),
         "flag only (kept)",
         "may be genuine events (promotions, holidays) or data errors; removal would bias"
         " the promo effect. Elasticity has a robustness check excluding them.")

    # 7. Derived price / promotion variables
    df["discount_pct"] = (1 - df["price"] / df["base_price"]).clip(lower=0)
    df["on_discount"] = (df["discount_pct"] > 0.005).astype(int)
    df["any_promo"] = ((df["on_discount"] == 1) | (df["feature"] == 1) | (df["display"] == 1)).astype(int)
    df["log_units"] = np.log(df["units"])
    df["log_price"] = np.log(df["price"])
    df["log_base_price"] = np.log(df["base_price"])
    df["log_discount_ratio"] = df["log_price"] - df["log_base_price"]  # <= 0

    # 8. Attach store and product attributes
    stores = stores.rename(columns={"store_id": "store_num"})
    dup_st = stores["store_num"].duplicated(keep=False)
    if dup_st.any():
        n_aff = df["store_num"].isin(stores.loc[dup_st, "store_num"]).sum()
        _log(log, "store lookup: same store_id listed twice with different segment", n_aff,
             "keep first listing (MAINSTREAM); rows kept",
             f"stores {sorted(stores.loc[dup_st, 'store_num'].unique().tolist())} appear twice;"
             " a naive join would duplicate their transactions")
        stores = stores.drop_duplicates("store_num", keep="first")
    stores = stores[["store_num", "store_name", "address_state_prov_code", "seg_value_name",
                     "sales_area_size_num", "avg_weekly_baskets"]].rename(
        columns={"address_state_prov_code": "state", "seg_value_name": "store_segment"})
    products = products[["upc", "description", "manufacturer", "category", "sub_category", "product_size"]]
    df = df.merge(stores, on="store_num", how="left").merge(products, on="upc", how="left")
    _log(log, "rows without store attributes", df["store_segment"].isna().sum(), "keep",
         "store lookup joined; missing values would be left as NaN")

    # Drop same-week outcome variables' leakage risk is handled in feature_engineering.
    _log(log, "final clean rows", len(df), "-", f"{n0 - len(df)} rows removed in total")
    return df, pd.DataFrame(log)


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of possible store-product-weeks that actually appear in the data.

    A missing week can mean: zero demand, out-of-stock, product not ranged in the
    store, or a data gap. The data cannot distinguish these (stockout problem).
    """
    weeks = df["week"].nunique()
    pairs = df.groupby(["upc", "description"]).agg(
        stores=("store_num", "nunique"), rows=("units", "size")).reset_index()
    pairs["possible_rows"] = pairs["stores"] * weeks
    pairs["coverage"] = pairs["rows"] / pairs["possible_rows"]
    return pairs.sort_values("coverage")
