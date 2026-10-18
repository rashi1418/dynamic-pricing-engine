"""Demand-response function + constrained price optimization.

Demand response (constant elasticity, consistent with the log-log model):

    Q(P) = Q0 * (P / P0) ** E

    Q0 = baseline demand at the reference price P0 (from the XGBoost forecast,
         with promotions switched off and price = current regular price)
    P0 = current regular price (demand-weighted across stores)
    E  = decision elasticity (regular-price elasticity, see elasticity.py)

Why combine the two models this way? The forecast model answers "how many units
next week at today's price?" (level); the elasticity answers "by what % does that
change if the regular price moves by x%?" (slope). XGBoost alone is not used for
the slope: its price response is learned mostly from promotional weeks, is not
constrained to be monotone and is confounded (see notebook 04).

Objective (cost is NOT in the data -> ASSUMED unit cost C = (1 - margin) * P0):

    Revenue  R(P) = P * Q(P)
    Profit   Pi(P) = (P - C) * Q(P)

Constraints:
    price band      : (1 - d) * P0 <= P <= (1 + d) * P0      (d = 20% by default)
    volume guardrail: Q(P) >= (1 - v) * Q0                   (v = 10% by default)
                      <=> P <= P0 * (1 - v) ** (1 / E)        for E < 0
    cost floor      : P > C
The volume guardrail exists because a single-product profit objective ignores
store traffic, basket effects and long-run customer loss (see the Lerner-condition
discussion in notebook 04): without it almost every product hits the +20% cap.

Useful closed forms for constant elasticity (used as a sanity check):
    * Revenue: R(P) ~ P^(1+E). E < -1 -> revenue falls with price (optimum at lower bound);
      E > -1 -> revenue rises with price (optimum at upper bound). Never interior.
    * Profit : for E < -1 the unconstrained optimum is P* = C * E / (1 + E)
      (the classic markup rule); for -1 <= E < 0 profit rises with price -> upper bound.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from . import config


def demand_at_price(P, Q0, P0, E):
    return Q0 * (np.asarray(P, dtype=float) / P0) ** E


def objective_value(P, Q0, P0, E, unit_cost=None, objective="profit"):
    Q = demand_at_price(P, Q0, P0, E)
    if objective == "revenue":
        return P * Q
    return (P - unit_cost) * Q


def analytic_optimum(E, unit_cost, objective="profit"):
    """Unconstrained optimum (None if it does not exist)."""
    if objective == "profit" and E < -1:
        return unit_cost * E / (1 + E)
    return None


def optimize_price(Q0, P0, E, unit_cost=None, objective="profit",
                   max_change=config.MAX_PRICE_CHANGE, max_volume_loss=config.MAX_VOLUME_LOSS,
                   min_price=None, max_price=None):
    """Bounded scalar optimization + grid cross-check. Returns a dict of results and flags."""
    lo, hi = P0 * (1 - max_change), P0 * (1 + max_change)
    band_hi = hi
    if max_volume_loss is not None and E < 0:
        hi = min(hi, P0 * (1 - max_volume_loss) ** (1 / E))
    if min_price is not None:
        lo = max(lo, min_price)
    if max_price is not None:
        hi = min(hi, max_price)
    infeasible = False
    if objective == "profit" and unit_cost is not None:
        lo = max(lo, unit_cost * 1.001)  # never price below cost
    if lo >= hi:
        # cost floor above the allowed band (e.g. clearance pricing): no price is both
        # profitable and within the business rules -> charge the highest allowed price
        infeasible, lo = True, hi

    f = lambda P: -objective_value(P, Q0, P0, E, unit_cost, objective)
    if infeasible:
        res = type("R", (), {"x": hi, "success": True})()
    else:
        res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    P_opt = float(res.x)
    # bounded Brent never evaluates exactly at the bounds -> snap if a bound is better
    for b in (lo, hi):
        if f(b) <= f(P_opt):
            P_opt = b

    grid = np.linspace(lo, hi, 401) if hi > lo else np.array([hi])
    vals = -f(grid)
    P_grid = float(grid[np.argmax(vals)])
    tol = 1e-3 * P0

    out = {
        "current_price": P0, "recommended_price": P_opt,
        "price_change_pct": P_opt / P0 - 1,
        "elasticity": E, "unit_cost_assumed": unit_cost,
        "demand_current": Q0, "demand_recommended": float(demand_at_price(P_opt, Q0, P0, E)),
        "revenue_current": P0 * Q0,
        "revenue_recommended": float(objective_value(P_opt, Q0, P0, E, objective="revenue")),
        "lower_bound": lo, "upper_bound": hi,
        "at_lower_bound": abs(P_opt - lo) < tol, "at_upper_bound": abs(P_opt - hi) < tol,
        "optimizer_matches_grid": abs(P_opt - P_grid) <= (hi - lo) / 400 + tol,
        "optimizer_success": bool(res.success),
    }
    if unit_cost is not None:
        out["profit_current"] = (P0 - unit_cost) * Q0
        out["profit_recommended"] = float(objective_value(P_opt, Q0, P0, E, unit_cost, "profit"))
    a = analytic_optimum(E, unit_cost, objective)
    out["analytic_unconstrained_optimum"] = a
    out["solution_type"] = ("lower bound" if out["at_lower_bound"] else
                            "upper bound" if out["at_upper_bound"] else "interior")
    out["binding_constraint"] = (
        "cost floor above band (infeasible -> max allowed price)" if infeasible else
        "-price band" if out["at_lower_bound"] else
        ("+price band" if abs(hi - band_hi) < tol else "volume guardrail") if out["at_upper_bound"]
        else "none (interior optimum)")
    return out


def objective_curve(Q0, P0, E, unit_cost=None, span=0.35, n=200) -> pd.DataFrame:
    P = np.linspace(P0 * (1 - span), P0 * (1 + span), n)
    Q = demand_at_price(P, Q0, P0, E)
    df = pd.DataFrame({"price": P, "demand": Q, "revenue": P * Q})
    if unit_cost is not None:
        df["profit"] = (P - unit_cost) * Q
    return df


def sanity_checks(r: dict) -> dict:
    """Automatic checks that the recommendation is economically coherent."""
    E = r["elasticity"]
    curve = objective_curve(r["demand_current"], r["current_price"], E, r["unit_cost_assumed"])
    checks = {
        "elasticity_negative": E < 0,
        "demand_decreasing_in_price": bool(np.all(np.diff(curve["demand"]) < 0)),
        "optimizer_matches_grid": r["optimizer_matches_grid"],
        "within_constraints": r["lower_bound"] - 1e-9 <= r["recommended_price"] <= r["upper_bound"] + 1e-9,
        "above_unit_cost": r["unit_cost_assumed"] is None or r["recommended_price"] > r["unit_cost_assumed"],
    }
    a = r["analytic_unconstrained_optimum"]
    if a is not None:  # the constrained optimum must equal the analytic one clipped to the bounds
        expected = min(max(a, r["lower_bound"]), r["upper_bound"])
        checks["matches_closed_form"] = abs(expected - r["recommended_price"]) < 1e-3 * r["current_price"]
    elif E >= -1:
        checks["matches_closed_form"] = r["at_upper_bound"]  # profit increasing in P
    checks["all_passed"] = all(checks.values())
    return checks


def interpret(r: dict, margin=config.ASSUMED_GROSS_MARGIN) -> str:
    """Plain-English explanation of one recommendation."""
    E, ch = r["elasticity"], r["price_change_pct"]
    kind = "elastic (|E| > 1)" if E < -1 else "inelastic (|E| < 1)"
    txt = (f"Estimated price elasticity is {E:.2f}: demand is {kind}. A 1% price rise is "
           f"associated with a {abs(E):.2f}% drop in units. ")
    if r["solution_type"] == "interior":
        txt += (f"With the ASSUMED {margin:.0%} gross margin the profit-maximizing price is "
                f"C*E/(1+E) = {r['recommended_price']:.2f}, inside the +/-{config.MAX_PRICE_CHANGE:.0%} band. ")
    elif r["binding_constraint"] == "volume guardrail":
        txt += (f"At the ASSUMED {margin:.0%} margin, a higher price would raise expected profit, but the "
                f"increase is capped where expected units fall by {config.MAX_VOLUME_LOSS:.0%} (volume guardrail). "
                "The less elastic the product, the larger the price move the guardrail allows. ")
    elif r["solution_type"] == "upper bound":
        txt += ("Demand is not very price-sensitive, so every increase in the allowed band raises "
                "expected profit and the optimizer stops at the +20% band. The guardrail, not the model, "
                "determines the size of the move - treat this as 'test a price increase', not as a precise target. ")
    else:
        txt += ("Demand is so price-sensitive that a lower price raises expected profit even at the assumed "
                "margin; the optimizer stops at the -20% guardrail. ")
    txt += (f"Expected weekly units change {r['demand_recommended'] / r['demand_current'] - 1:+.1%}, "
            f"price {ch:+.1%}.")
    return txt


# --------------------------------------------------------------------------
# Sanity check: what does the forecasting model itself 'think' about price?
# --------------------------------------------------------------------------
def model_price_response(model, rows: pd.DataFrame, multipliers=np.linspace(0.8, 1.2, 9)) -> pd.DataFrame:
    """Average XGBoost prediction when the regular price of `rows` is scaled by each
    multiplier (promotions off). Used only as a diagnostic, not for decisions."""
    from .feature_engineering import set_price_scenario
    from .forecasting import predict_units
    out = []
    for m in multipliers:
        p = rows["recent_base_price"] * m
        r = set_price_scenario(rows, price=p, base_price=p, feature=0, display=0)
        out.append({"multiplier": m, "mean_pred_units": float(predict_units(model, r).mean())})
    df = pd.DataFrame(out)
    df["relative_demand"] = df["mean_pred_units"] / df.loc[(df["multiplier"] - 1).abs().idxmin(), "mean_pred_units"]
    return df


# --------------------------------------------------------------------------
# Dynamic engine: gradual price paths
# --------------------------------------------------------------------------
def price_path(target, start=1.0, max_step=config.MAX_WEEKLY_CHANGE, weeks=13):
    """Move a price multiplier from `start` toward `target`, at most `max_step`
    (relative) per week. Returns the multiplier for each of the next `weeks` weeks."""
    path, m = [], start
    for _ in range(weeks):
        if target > m:
            m = min(target, m * (1 + max_step))
        elif target < m:
            m = max(target, m * (1 - max_step))
        path.append(m)
    return np.array(path)


def weeks_to_target(target, start=1.0, max_step=config.MAX_WEEKLY_CHANGE):
    if abs(target - start) < 1e-9:
        return 0
    return int(np.ceil(abs(np.log(target / start)) / np.log(1 + max_step) - 1e-9))
