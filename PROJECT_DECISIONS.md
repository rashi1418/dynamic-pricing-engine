# Project Decisions: Why We Did It This Way
*(Dynamic Pricing Engine for Retail Demand Optimization)*

Each decision is written as **Decision → Reason → Alternative → Why not the alternative → Trade-off → Limitation**.
Numbers quoted come from `reports/results/` (produced by `run_pipeline.py`).

---

## Data

### 1. Why dunnhumby "Breakfast at the Frat"?
- **Reason:** It contains both the actual shelf price and the regular (base) price, plus separate feature, display and TPR flags. So we can separate regular-price variation from promotional variation and control for promotion visibility.
- **Alternative:** Dominick's Finer Foods (bigger, even has margins); M5/Walmart (famous); Favorita, Rossmann.
- **Why not:** Dominick's host was unreachable from the build environment, so an end-to-end run could not be verified. M5 has no promotion flags and needs a Kaggle login. Favorita and Rossmann have **no price**, so no elasticity is possible.
- **Trade-off:** Only 55 products and 4 categories; data from 2009-2011.
- **Limitation:** Conclusions are about this retailer and these categories only.

### 2. Why download from a GitHub mirror?
- **Reason:** dunnhumby's download is behind a web form, which would break reproducibility. The mirror is pinned to a commit and verified by MD5. The same checksum appears on a second independent mirror.
- **Alternative:** Ask the user to download manually.
- **Why not:** Kept as the fallback (data/README.md). Automation is better for a reproducible project.
- **Trade-off:** Depends on a third party keeping the file online.
- **Limitation:** We do not redistribute the data; users must respect dunnhumby's terms.

### 3. Why flag outliers instead of deleting them?
- **Reason:** Extreme weeks are mostly real (promotions, Super Bowl). Deleting them deletes the variation that identifies promotion and price effects, and biases the forecast downward in event weeks.
- **Alternative:** IQR/z-score deletion, winsorisation.
- **Why not:** No evidence they are errors. We ran a robustness check instead: elasticities without flagged rows correlate 0.964 with the main estimates.
- **Trade-off:** A few genuine data errors may remain.

### 4. Why missing weeks are NaN, not 0
- **Reason:** The data has no rows for weeks without sales. Such a week could be zero demand, a stockout, the product not being stocked, or a data gap. Filling 0 would teach the model that demand collapses whenever the product was unavailable.
- **Alternative:** Fill zeros (common in Kaggle solutions).
- **Why not:** It encodes an assumption we cannot verify.
- **Trade-off:** We only forecast "units given that the product sells", and elasticities exclude zero-sales weeks (selection bias, probably toward zero).

### 5. Why deduplicate the store lookup (keep first listing)?
- **Reason:** Stores 4503 and 17627 are listed twice with different segments. A naive join duplicated 13,693 transactions and would inflate every downstream number.
- **Alternative:** Drop those stores.
- **Why not:** Their sales data is fine; only a store attribute is ambiguous.
- **Limitation:** Their segment label may be wrong. The label is only a minor forecasting feature.

## Forecasting

### 6. Why weekly lags 1, 2, 3, 4, 8, 13, 52 (not 7/14/28)?
- **Reason:** The data is weekly. Lag 1 = last week, 4 ≈ a month, 13 = a quarter, 52 = same week last year.
- **Alternative:** Daily-style lags from the project template.
- **Why not:** A lag of 7 *weeks* would be arbitrary here.

### 7. Why use target-week price and promotion as features?
- **Reason:** Retailers plan prices and promotions in advance (price files, ad circulars), so they are known at forecast time. They are also necessary to ask "what if price = P?".
- **Alternative:** Exclude them to be ultra-conservative about leakage.
- **Why not:** They are not leakage. Leakage is using information unavailable at prediction time (e.g. same-week `visits`, `spend`), and those are excluded.
- **Limitation:** If plans change after the forecast is made, the forecast is conditional on the wrong inputs.

### 8. Why XGBoost (and LightGBM)?
- **Reason:** Tabular, mixed numeric/categorical data with interactions (promotion × category × store). Gradient-boosted trees are the strongest general-purpose tool here, fast on 1 CPU, and handle missing lags natively.
- **Alternative:** LSTM/Transformer, ARIMA per series, Prophet, linear models.
- **Why not:** ~3,900 short, gappy series (≤156 points each). Per-series ARIMA cannot share information across products and handles promotions badly. Deep sequence models need more data and tuning for little expected gain on tabular features, and are harder to explain. LightGBM is included to show the result is not library-specific (test WMAPE 0.287 for both).
- **Trade-off:** Trees cannot extrapolate beyond observed feature ranges, which matters for price what-ifs (decision 18).

### 9. Why walk-forward validation instead of random K-fold?
- **Reason:** A model is always trained on the past and predicts the future. Random K-fold puts the same weeks (and neighbouring weeks) in train and validation. Rolling features of a validation row then overlap training targets, so the scores are optimistic.
- **Alternative:** `sklearn.TimeSeriesSplit`.
- **Why not directly:** It splits by row position, which can cut a week in half. We apply the same expanding-window idea, but split on *weeks*.
- **Trade-off:** Early stopping uses each fold's validation block, so CV scores are slightly optimistic. The 2011 test year is never used for any choice.

### 10. Why a seasonal-naive baseline (and why did it lose)?
- **Reason:** A model must beat the obvious answer. For weekly retail data "same week last year" is the natural seasonal benchmark.
- **Result:** It was the worst model (WMAPE 0.587 vs naive 0.433): demand follows a promotion calendar that changes every year. This is a finding about the data, and it explains why price and promotion features matter.

### 11. Why log1p target + a bias correction (instead of Poisson)?
- **Reason:** Units are skewed (median 10, max 1,800). The log turns the loss into roughly relative error. But expm1 of a mean log under-forecasts (Jensen's inequality): bias was −5.5%. A factor Σy/Σŷ from the last training fold (1.047) reduced the bias to −1.0% with no WMAPE change.
- **Alternative tried during development:** XGBoost `count:poisson` on raw units. Last CV fold WMAPE 0.296 (vs 0.293 for log1p), bias +0.1%; test WMAPE 0.292 vs 0.287, bias −0.6%.
- **Why not:** Slightly worse WMAPE; the simple correction closed most of the bias gap.
- **Trade-off:** One global correction factor, so not every product/store is unbiased.

### 12. Why WMAPE as the primary metric?
- **Reason:** Volume-weighted, scale-free and robust to small actuals. MAPE is dominated by 1-2 unit weeks (MAPE ≈ 0.55 while WMAPE ≈ 0.29).
- **Alternative:** RMSE (reported; dominated by big stores), MAPE (reported, for completeness).

### 13. Why SHAP?
- **Reason:** Consistent, additive, local and global explanations for tree models (TreeExplainer is exact and fast).
- **Alternative:** Built-in gain importance (biased toward high-cardinality features), permutation importance.
- **Limitation:** SHAP explains the *model*, not the world. A large SHAP value for price is not a causal price effect.

## Elasticity

### 14. Why a log-log model?
- **Reason:** In `log Q = β log P`, β is directly the elasticity (percent-for-percent), assumed constant over the range. It matches multiplicative demand, compresses skew, and makes `Q(P) = Q0 (P/P0)^β` exactly consistent with the estimate.
- **Alternative:** Linear `Q ~ P` (elasticity varies along the curve), semi-log, non-parametric.
- **Trade-off:** Constant elasticity is a local approximation. Fine within ±20%; dangerous far outside the observed range.

### 15. Why not simple correlation / naive OLS?
- **Reason:** Correlation mixes price with everything that moves with price. Naive slopes had median −2.07, and four products even had *positive* slopes (private-label pretzels +1.8), which would say "raise price, sell more".
- **Evidence:** On held-out 2011 price changes the naive elasticities overstate the true response about 2.5× (calibration slope 0.39).

### 16. Why use promotion events to identify the elasticity (M8)?
- **Reason:** Promotions are the main source of price variation, and also the main confounder: a promoted week usually has a price cut *plus* a feature ad and/or display. M8 turns promotions into clean identifying variation:
  - keep only no-promotion weeks and **pure temporary price reductions (no ad, no display)**, so the price cut is not bundled with advertising;
  - **store FE** remove permanent store differences;
  - **week FE** remove the chain's promotion timing and every chain-wide shock, so identification comes from *stores running different discounts in the same week* (≈26% of discount variance);
  - log(base price) enters separately, so the coefficient is the response to a *temporary* price change;
  - last week's discount controls for pantry loading.
- **Alternative:** Plain controls on all weeks (M4), regular-price variation only (M5), an instrument (M7).
- **Why not:** M4 is very similar in value and slightly better in held-out MAE, but it mixes ad/display weeks into the sample and relies on dummies to separate them. M5 uses rarer regular-price changes. M7 failed (decision 17).
- **Trade-off:** Fewer observations (feature/display weeks dropped); SEs larger (median ≈0.19).
- **Limitation:** Local store-week confounding remains (a store discounting because of local demand). Identification comes from price *cuts*; increases are extrapolated.
- **Evidence:** On 17,842 real 2011 price-cut events, the shrunk M8 elasticities have calibration slope ≈1.0 (right magnitude), estimated only on 2009-10.

### 17. Why was the promotion-calendar instrument (M7) rejected?
- **Idea:** ≈74% of discount-depth variance is chain-wide, so promotions are set centrally. Instrument a store's price with the same product's average discount in the other stores that week.
- **Relevance:** Very strong (median first-stage F ≈ 1,900; 3 of 55 products weak).
- **Validity (exclusion restriction):** Fails. Its median estimate (−2.05) reproduces the naive OLS (−2.07), and for Frescetta pizzas it gives −3.4 to −4.0 vs ≈ −0.8 from the fully controlled model. The chain schedules promotions into high-demand weeks, so the calendar is correlated with demand. Out of sample it is worse than assuming no price effect (MAE skill −21.8%).
- **Lesson:** A strong instrument is not a valid instrument. The IV is kept in the project as a documented negative result.

### 18. Why empirical-Bayes shrinkage instead of a category fallback?
- **Reason:** Some products have few TPR events and noisy estimates (Mike-sell's Pretzel Sticks +1.55 ± 0.77). Replacing them with a category average throws away product information. Shrinkage keeps a **product-level estimate for every product**, pulled toward the category mean in proportion to its noise: `shrunk = μ + τ²/(τ²+se²)·(β̂ − μ)`, with τ² (true between-product spread) from DerSimonian-Laird.
- **Alternative:** Hard fallback rule; full hierarchical Bayesian model.
- **Why not:** A hard rule is all-or-nothing. A full Bayesian model is better but heavier; listed as a future improvement.
- **Evidence:** Shrinkage moved the held-out calibration slope from 0.65 (raw M8) to 1.02.
- **Limitation:** Assumes products in a category share a common distribution of elasticities; τ² is estimated, and that uncertainty is not propagated.

### 19. Why store-segment elasticities, and why are they small?
- **Reason:** A pricing engine can price value, mainstream and upscale stores differently. M8 with segment interactions found 18 of 93 estimated deviations significant (upscale stores less price-sensitive). Deviations are shrunk toward 0 within category, so only consistent differences survive.
- **Result:** After shrinkage the segment adjustments are small (a few hundredths to a tenth). A segment adjustment may refine but never flip a usable product's sign (capped at −0.05).

### 20. Why not use XGBoost's price response as the elasticity?
- **Reason:** When regular price was scaled 0.8→1.2 inside XGBoost, demand was monotone decreasing for only 43% of products. Trees learn price mainly from promotion weeks and predict a constant beyond the last split.
- **Decision:** XGBoost gives the **level** Q0; the elasticity gives the **slope**.

### 21. Minimum data requirements and "usable" elasticities
- ≥500 store-weeks, ≥10 distinct prices, within-store-product SD of log price ≥ 0.03, all fixed in `config.py` before estimation. After shrinkage, an elasticity ≥ −0.1 is not usable: 4 products are therefore not priced and are flagged for a controlled test.

## Optimization and dynamic engine

### 22. Why optimize at all (instead of "elastic → cut price")?
- **Reason:** The best price depends on elasticity, cost and constraints together. An optimizer makes the trade-off explicit, reproducible and checkable.

### 23. Why profit with an assumed margin (not revenue)?
- **Reason:** With constant elasticity, revenue ∝ P^(1+E) is monotone, so the revenue optimum is *always* a bound. A retailer also cares about margin.
- **Trade-off:** Cost is not in the data. The ASSUMED 30% margin is explicit, and sensitivity is shown (simulated gain +70% at 20% margin, +21% at 40%; the price schedule itself barely changes because constraints bind).

### 24. Why ±20% price band?
- **Reason:** Standard commercial guardrail. Customers notice big jumps, and the log-log model is only trustworthy near observed prices.

### 25. Why a volume guardrail (units may fall at most 10%)?
- **Reason:** 32 of 51 usable products have |E| ≤ 1, for which profit keeps rising with price, and without the guardrail every product hits +20%. This is the Lerner condition (optimal margin = −1/E ≈ 67% for the median elastic product). The single-product model omits store traffic, basket effects, competitor response and long-run customer loss; the guardrail is an explicit stand-in.
- **Trade-off:** 94 of 141 recommendations sit on the guardrail and 47 on the band, so constraints set the size of the move; the data sets the ranking.

### 26. Why a "dynamic" engine (weekly targets, quarterly re-estimation, ≤5% per week)?
- **Reason:** Demand and price sensitivity drift; a pricing system should use new data as it arrives and avoid shocking customers with large jumps. The engine recomputes targets weekly, re-estimates elasticities every 13 weeks on all past data, and moves prices at most 5% per week.
- **Alternative:** One static price change; real-time pricing.
- **Why not real-time:** Weekly scanner data and shelf-label logistics; regular prices don't change daily in grocery.
- **Honest result:** In the 2011 replay the dynamic engine (+34.3% simulated profit) does **not** beat a one-time static change (+35.0%). The gradual path delays part of the gain and the segment differences are small. The value is control and adaptivity, not extra simulated profit. With constant elasticity the optimal markup doesn't depend on the demand level, so weekly targets only move when elasticities are re-estimated.

### 27. Why SciPy `minimize_scalar(method="bounded")`?
- **Reason:** One decision variable, a smooth objective, simple bounds. Bounded Brent is the right-sized tool. Two edge cases are handled explicitly: Brent never evaluates exactly at the bounds (the code compares against both bounds), and the cost floor can lie above the allowed band for clearance items (no feasible profitable price, so the engine charges the highest allowed price and flags it).

### 28. Why the sanity checks?
- **Reason:** Optimizers return *something* even if the inputs are nonsense. Each recommendation checks: negative elasticity, demand decreasing in price, optimizer = grid argmax, constraints respected, price above cost, and agreement with the closed-form optimum. All 141 product × segment recommendations pass.

### 29. Why exclude discontinued / phased-out products?
- **Reason:** Colgate Wisp stopped selling in March 2011; three Quaker cereals sold in only 2-7 of ~77 stores/week in the final month. A recommendation for a product leaving the shelf is meaningless (rule: average stores/week in the last 4 weeks ≥ 25% of usual).

## Backtesting

### 30. Why two kinds of historical backtest?
- **Backtest 1, price-response validation:** this is the part that can be checked against reality. Elasticities from 2009-10 predict the sales lift of 17,842 real 2011 store-weeks with a pure price cut. Lifts are measured against the store's own recent non-promoted weeks and against stores that did *not* promote that week (removing holidays and other chain-wide shocks). Metrics: MAE skill vs "no price effect" and calibration slope.
- **Backtest 2, walk-forward replay:** the engine is run as it would have been in 2011 (quarterly re-estimation using only past data, weekly paths), and outcomes are simulated by adjusting observed units by `m^E`. This is a **counterfactual simulation**: nobody observed 2011 customers at the engine's prices.

### 31. Why anchor the simulation to observed demand?
- **Reason:** Keeping actual 2011 units (with their real shocks) in both scenarios isolates the effect of the price change.
- **Trade-off:** It assumes each week's shock would have been the same under the new price.

### 32. Why stress tests?
- **Reason:** A schedule judged by the elasticity that designed it looks good by construction. It is re-judged with 2011-re-estimated elasticities, CI bounds and a different model (controlled OLS): the simulated profit gain ranges +27.9% to +39.6%. The sign is robust; the size depends on the unobserved margin.

### 33. Why not claim causal impact?
- **Reason:** The elasticity is observational (local confounding, selection), validated on price cuts but applied to increases, and the simulation holds competitors, promotions and customer behaviour fixed. Only a controlled store-level price test measures the causal effect.
