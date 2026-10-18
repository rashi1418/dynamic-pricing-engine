"""Demand forecasting: baselines, walk-forward validation, XGBoost and LightGBM.

Forecasting task: one-week-ahead unit sales for every (store, product, week)
that has a recorded sale. Price and promotion of the target week are treated as
known (planned) inputs; demand history is strictly lagged.

Target transform: models are trained on log1p(units). Unit sales are right-skewed
(median 10, max 1,800); the log makes the loss focus on relative errors so that a
few huge stores/promotions do not dominate the fit. Predictions are mapped back
with expm1 and all metrics are computed on the original unit scale.

Retransformation bias: expm1(E[log1p y]) < E[y] (Jensen's inequality), so a model
trained on log1p under-forecasts total volume (~ -5% here). We correct this with a
single multiplicative factor = sum(actual) / sum(predicted) measured on the LAST
walk-forward validation fold (training data only), then apply it to test forecasts.
"""
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from . import config
from .evaluation import all_metrics
from .feature_engineering import feature_columns

warnings.filterwarnings("ignore", message=".*eval_set.*is deprecated.*")

XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.05, max_depth=8, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
    enable_categorical=True, max_cat_to_onehot=1, random_state=config.SEED,
    early_stopping_rounds=100, eval_metric="rmse", n_jobs=-1,
)
LGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.05, num_leaves=127, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=config.SEED, n_jobs=-1, verbose=-1,
)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def train_test_masks(df):
    obs = df["observed"] == 1
    return obs & (df["week"] <= config.TRAIN_END), obs & (df["week"] >= config.TEST_START)


def walk_forward_folds(df, n_folds=config.N_CV_FOLDS, val_weeks=config.CV_VAL_WEEKS):
    """Expanding-window folds inside the training period (like TimeSeriesSplit,
    but split on *weeks*, so a week is never partly in train and partly in validation).

    fold k: train = all weeks before the validation block, validate = next 13 weeks.
    """
    weeks = np.sort(df.loc[df["week"] <= config.TRAIN_END, "week"].unique())
    folds = []
    for k in range(n_folds, 0, -1):
        val_end = len(weeks) - (k - 1) * val_weeks
        val_start = val_end - val_weeks
        tr_w, va_w = weeks[:val_start], weeks[val_start:val_end]
        obs = df["observed"] == 1
        folds.append((obs & df["week"].isin(tr_w), obs & df["week"].isin(va_w),
                      pd.Timestamp(va_w[0]), pd.Timestamp(va_w[-1])))
    return folds


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def _fallback(df, cols):
    out = pd.Series(np.nan, index=df.index)
    for c in cols:
        out = out.fillna(df[c])
    return out.fillna(0)


def baseline_predictions(df) -> dict:
    """Simple benchmarks every ML model must beat.

    Missing lags (a gap week, or no history one year ago) fall back to the most
    recent available information, so every row gets a prediction.
    """
    return {
        "Naive (last week)": _fallback(df, ["units_lag_1", "units_rollmean_4", "units_expanding_mean"]),
        "Seasonal naive (lag 52)": _fallback(df, ["units_lag_52", "units_rollmean_4", "units_expanding_mean"]),
        "Moving average (4 weeks)": _fallback(df, ["units_rollmean_4", "units_expanding_mean"]),
    }


# --------------------------------------------------------------------------
# ML models
# --------------------------------------------------------------------------
def _xy(df, mask):
    X = df.loc[mask, feature_columns()]
    y = df.loc[mask, "units"].astype(float)
    return X, y


def fit_xgb(df, tr, va=None, n_estimators=None):
    X, y = _xy(df, tr)
    params = XGB_PARAMS.copy()
    if va is None:
        params.pop("early_stopping_rounds")
        params["n_estimators"] = n_estimators
        m = xgb.XGBRegressor(**params)
        m.fit(X, np.log1p(y), verbose=False)
    else:
        Xv, yv = _xy(df, va)
        m = xgb.XGBRegressor(**params)
        m.fit(X, np.log1p(y), eval_set=[(Xv, np.log1p(yv))], verbose=False)
    return m


def fit_lgb(df, tr, va=None, n_estimators=None):
    X, y = _xy(df, tr)
    params = LGB_PARAMS.copy()
    if va is None:
        params["n_estimators"] = n_estimators
        m = lgb.LGBMRegressor(**params)
        m.fit(X, np.log1p(y))
    else:
        Xv, yv = _xy(df, va)
        m = lgb.LGBMRegressor(**params)
        m.fit(X, np.log1p(y), eval_set=[(Xv, np.log1p(yv))],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def predict_units(model, X) -> np.ndarray:
    return np.clip(np.expm1(model.predict(X[feature_columns()])), 0, None)


def best_iter(model):
    if isinstance(model, xgb.XGBRegressor):
        return int(model.best_iteration) + 1
    return int(model.best_iteration_)


def run_cv_fold(df, fold: int):
    """Evaluate baselines + ML models on ONE walk-forward fold (1 = earliest).

    Note: the fold's validation block is also used for early stopping, so CV
    scores of the ML models are slightly optimistic. The final 2011 test set is
    never used for any choice and gives the unbiased estimate.
    """
    tr, va, v0, v1 = walk_forward_folds(df)[fold - 1]
    y = df.loc[va, "units"].values
    rows, iters, calib = [], {}, {}
    for name, p in baseline_predictions(df).items():
        rows.append({"fold": fold, "val_start": v0.date(), "val_end": v1.date(), **all_metrics(y, p[va].values, name)})
    for name, fit in (("XGBoost", fit_xgb), ("LightGBM", fit_lgb)):
        m = fit(df, tr, va)
        iters[name] = best_iter(m)
        p = predict_units(m, df.loc[va])
        calib[name] = float(y.sum() / p.sum())
        rows.append({"fold": fold, "val_start": v0.date(), "val_end": v1.date(),
                     **all_metrics(y, p, name), "best_iteration": best_iter(m)})
    return pd.DataFrame(rows), {"best_iteration": iters, "bias_correction": calib}


def combine_cv(fold_infos: list[dict]) -> dict:
    """Number of trees = mean best iteration over folds; bias correction = LAST fold
    (closest to the test period)."""
    names = fold_infos[0]["best_iteration"].keys()
    return {"n_estimators": {n: int(np.mean([f["best_iteration"][n] for f in fold_infos])) for n in names},
            "bias_correction": fold_infos[-1]["bias_correction"]}


def run_walk_forward_cv(df):
    """All folds in one call (used by notebook 02)."""
    results = [run_cv_fold(df, k) for k in range(1, config.N_CV_FOLDS + 1)]
    return pd.concat([r[0] for r in results], ignore_index=True), combine_cv([r[1] for r in results])


def fit_final_and_test(df, cv_info: dict):
    """Refit on the full training period with the CV-chosen number of trees and
    evaluate once on the untouched 2011 test period."""
    n_iters, calib = cv_info["n_estimators"], cv_info["bias_correction"]
    tr, te = train_test_masks(df)
    y = df.loc[te, "units"].values
    results, preds = [], {}
    for name, p in baseline_predictions(df).items():
        preds[name] = p[te].values
        results.append(all_metrics(y, preds[name], name))
    models = {"XGBoost": fit_xgb(df, tr, n_estimators=n_iters["XGBoost"]),
              "LightGBM": fit_lgb(df, tr, n_estimators=n_iters["LightGBM"])}
    for name, m in models.items():
        raw = predict_units(m, df.loc[te])
        results.append(all_metrics(y, raw, f"{name} (no bias correction)"))
        preds[name] = raw * calib[name]
        results.append(all_metrics(y, preds[name], name))
    return models, pd.DataFrame(results).set_index("model"), preds


def fit_production_model(df, n_estimators):
    """Refit XGBoost on ALL observed weeks (train + test) - used only to
    forecast the next, unobserved week for the final price recommendation."""
    obs = df["observed"] == 1
    return fit_xgb(df, obs, n_estimators=n_estimators)
