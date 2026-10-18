# Project Audit

## Resume-bullet traceability
Each claim in the project description is implemented, executed, and backed by a file.

**Dynamic Pricing Engine for Retail Demand Optimization**
- Code: `run_pipeline.py --stage optimize`, `src/optimization.py`, `src/backtesting.py`.
- What makes it dynamic: weekly targets per product × store segment (141 recommendations), elasticities re-estimated every 13 weeks on all past data, and price moves of at most 5% per week (`price_path`, `dynamic_schedule`).
- Evidence: `business_recommendations.csv` (next-week price, target, weeks to target) and `dynamic_price_schedule.parquet` (2011 weekly path).

**"Built a demand forecasting pipeline for retail sales using XGBoost and LightGBM with lag and rolling feature engineering."**
- Code: `src/feature_engineering.py` (lags 1-4/8/13/52, rolling mean 4/8/13, rolling std, leakage-safe) and `src/forecasting.py`.
- Evidence: `model_comparison_test.csv`, `models/xgb_*.json`, `models/lgb_train_only.txt`, notebook 02.

**"Validated forecasts using time aware cross-validation and benchmarked accuracy against a seasonal naive baseline model."**
- Code: `walk_forward_folds` (4 expanding folds, split on weeks) plus an untouched 2011 test year.
- Evidence: `cv_results.csv`, `model_comparison_test.csv`. Seasonal naive WMAPE 0.587 vs XGBoost 0.288 (−51%).

**"Estimated product-level price elasticity via log-log regression, addressing confounding using promotion-based variation."**
- Code: `src/elasticity.py`, functions `estimate_promo_identified_product` (M8) and `decision_elasticities`.
- Product level: every one of the 55 products gets its own log-log estimate, with empirical-Bayes shrinkage instead of category substitution.
- Promotion-based variation: identification from temporary price reductions without ad or display, plus store and week fixed effects (removes the ad/display confound, store differences and the chain's promotion timing).
- Also tested: a promotion-calendar IV, rejected with documented evidence.
- Evidence: `promo_identified_elasticities.csv`, `product_elasticity_shrinkage.csv`, `decision_elasticity.csv`, notebook 03 §7-12.

**"Designed a constrained price optimization layer using SciPy to recommend optimal prices, validated via historical backtesting."**
- Code: `optimize_price` (SciPy `minimize_scalar` bounded, ±20% band, volume guardrail, cost floor, grid and closed-form checks).
- Backtesting, both estimated on 2009-10 only:
  1. elasticities scored against 17,842 real 2011 price changes (`price_response_validation.csv`, calibration ≈ 1.0);
  2. walk-forward replay of the engine over 2011 with stress tests (`backtest_summary.csv`).
- Wording note: "optimal" means profit-optimal under the stated constraints and an ASSUMED 30% margin. The replay outcomes are simulated, not observed.

## Requirement checklist
| Check | Status | Evidence |
|---|---|---|
| Dataset source documented | ✅ | `dataset_selection.md` (8 candidates compared), `data/README.md` |
| Dataset reproducible | ✅ | pinned URL + MD5; clean-room test: fresh download reproduced `panel_clean.parquet` identically |
| No fabricated data | ✅ | only observed dunnhumby data; the one non-observed quantity (unit cost) is labelled ASSUMED everywhere |
| No fabricated results | ✅ | every README number is in `reports/results/*.csv` or `metrics.json`; all 4 notebooks executed with 0 errors |
| No target leakage | ✅ | lags/rolling shifted ≥1 week on a complete calendar (asserted in notebook 02); same-week visits/HHS/spend excluded; all backtests estimated on data before the evaluated period |
| Chronological validation | ✅ | 4 walk-forward folds + untouched 2011 test |
| Seasonal naive baseline | ✅ | plus naive and moving-average baselines |
| XGBoost / LightGBM | ✅ | |
| MAE / RMSE / WMAPE | ✅ | plus MAPE (with caveat) and bias |
| SHAP | ✅ | `shap_importance.csv`, summary, dependence and waterfall plots |
| Price variation analysis | ✅ | `price_variation.csv`, minimum-data rules |
| Naive and controlled elasticity | ✅ | M1-M5 ladder, M6/M8 promotion-identified, M7 IV (rejected), clustered SEs, CIs, shrinkage, segments, robustness without outliers |
| Confounding discussion | ✅ | notebook 03, PROJECT_DECISIONS 15-19 |
| Optimization + constraints | ✅ | 141/141 sanity checks pass (`optimization_sanity_checks.csv`) |
| Demand / revenue / profit curves | ✅ | `opt_curves_*.png`, notebook 04, app |
| Backtesting | ✅ | price-response validation + walk-forward dynamic replay + stress tests |
| Counterfactual limitation | ✅ | notebook 04 §9, README Limitations |
| Docs | ✅ | README, PROJECT_DECISIONS (33 decisions), INTERVIEW_PREPARATION (22 topics), GLOSSARY |
| Streamlit app | ✅ | `streamlit.testing.AppTest`: all 47 priceable products, all stores for a product, promotion states, revenue objective, price path |

## Issues found and fixed during the build
1. Store lookup lists two stores twice → a naive join duplicated 13,693 rows → deduplicated and logged.
2. A merge turned categorical columns into integers, so XGBoost rejected scoring data → category levels persisted and re-applied (the same bug reappeared once when grouping by segment, and was fixed the same way).
3. Next-week baseline under-counted slow movers → extended to stores active in the last 4 weeks with a gap-aware calendar.
4. Discontinued/phased-out products produced near-zero baselines → excluded with explicit reasons.
5. Per-product profit % was meaningless when assumed historical profit was negative → absolute difference reported.
6. The textbook claim "deal elasticities exceed regular-price ones" was not supported by this data → corrected.
7. An exploratory backtest figure was not reproduced by the final pipeline → removed.
8. The promotion-calendar IV looked excellent (F ≈ 1,900) but reproduced the naive estimate and failed out of sample → rejected; documented as a negative result.
9. Missing store-segment estimates produced NaN elasticities and failed sanity checks → a missing segment uses the product estimate (deviation 0).
10. At a 20% assumed margin the cost floor exceeded the price band for clearance items, so the optimizer crashed → infeasible case handled and flagged.

## Known limitations (by design)
- Observational elasticities: local store-week confounding remains.
- Elasticities are identified on price cuts and applied to increases.
- Unit cost is assumed; zero-sales weeks are missing.
- Constant elasticity; no cross-price effects.
- Simulated (not observed) policy gains.
- In the 2011 replay, the dynamic engine does not beat a one-time static change in simulated profit.
