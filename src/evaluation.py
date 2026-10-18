"""Forecast accuracy metrics (all on the original unit scale)."""
import numpy as np
import pandas as pd


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def wmape(y, p):
    """Weighted MAPE = sum|y - p| / sum|y|.

    Robust to small / zero actuals (unlike MAPE) and weights errors by volume,
    which is what matters for inventory and revenue.
    """
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)))


def mape(y, p):
    """Classic MAPE on rows with y > 0. Reported for completeness only: it is
    dominated by low-volume rows (1 unit actual, 3 predicted = 200% error)."""
    m = y > 0
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]))


def bias(y, p):
    """(sum p - sum y) / sum y. Positive = over-forecasting."""
    return float((np.sum(p) - np.sum(y)) / np.sum(y))


def all_metrics(y, p, name=None) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    out = {"MAE": mae(y, p), "RMSE": rmse(y, p), "WMAPE": wmape(y, p),
           "MAPE": mape(y, p), "Bias": bias(y, p)}
    if name is not None:
        out = {"model": name, **out}
    return out


def comparison_table(results: list[dict], baseline="Seasonal naive (lag 52)") -> pd.DataFrame:
    t = pd.DataFrame(results).set_index("model")
    if baseline in t.index:
        t["WMAPE vs seasonal naive"] = (t["WMAPE"] / t.loc[baseline, "WMAPE"] - 1)
    return t
