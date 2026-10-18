# Interview Preparation: Dynamic Pricing Engine for Retail Demand Optimization

Each topic follows the same 8-part structure:
**1 What · 2 Why we used it · 3 Why not another method · 4 Assumptions · 5 What can go wrong · 6 Interviewer challenge · 7 Concise answer · 8 Deeper answer**

A note on honesty for interviews: say *"I built this as a learning project with AI assistance and studied every decision"*. Then prove
it by answering the follow-ups below. Every number here comes from `reports/results/`.

---

## A. Business Understanding
1. **What:** A dynamic pricing engine: every week, recommend a regular shelf price per product and store type (value / mainstream / upscale) that maximises (assumed) profit under business constraints, move prices toward it gradually, and explain why.
2. **Why:** Price is the strongest margin lever; rules of thumb ("+5% everywhere") ignore that products and stores differ in price sensitivity.
3. **Why not only forecasting:** A forecast answers "how much will sell at the planned price", not "what if we change it". The decision needs the *what-if*.
4. **Assumptions:** Products can be priced independently; ASSUMED 30% margin; customers respond to price as the elasticity says.
5. **What can go wrong:** Traffic and basket effects, competitor reaction, cannibalisation, biased elasticity, wrong cost.
6. **Challenge:** "Your engine raises almost every price. Is that believable?"
7. **Concise:** "Not literally. It identifies which products have room for an increase. The size is set mostly by a 10% volume guardrail, because a single-product profit model ignores store traffic."
8. **Deeper:** Optimal margin = −1/E (Lerner). 32 of 51 usable products have |E| ≤ 1, where profit keeps rising with price; without guardrails all would hit +20%. Either the retailer deliberately under-prices for traffic and competition, or the elasticities are biased toward zero, or both. The useful output is the *ranking*: Honey Nut Cheerios (E ≈ −0.1) gets +20%, DiGiorno Pepperoni (≈ −1.6) +6.8%. Next step: a store-level price test.

## B. Dataset
1. **What:** dunnhumby *Breakfast at the Frat*: 156 weeks, 77 stores, 55 products, 4 categories, weekly units with shelf price, base price and promo flags.
2. **Why:** It has both actual and regular price, so promotional and regular variation can be separated; feature/display flags allow controlling for promotion visibility.
3. **Why not Favorita/Rossmann/M5:** Favorita and Rossmann have no price; M5 has no promotion flags and needs Kaggle auth; Dominick's was not reachable from the build environment.
4. **Assumptions:** PRICE is the realised average price (verified: SPEND = PRICE × UNITS exactly); rows are representative of the stores.
5. **What can go wrong:** Missing zero-sales weeks; no cost; no competitor prices; old data (2009-2011).
6. **Challenge:** "How do you know the file wasn't tampered with?"
7. **Concise:** "The downloader verifies an MD5 checksum, and the same checksum appears on two independent mirrors."
8. **Deeper:** Pinned commit URL + checksum makes the pipeline reproducible. The data isn't committed (dunnhumby terms); manual instructions exist as a fallback. I also found data-quality issues myself: two duplicated stores in the lookup (would have duplicated 13,693 rows), 4,937 rows with price > base price, 5 zero-unit voids.

## C. EDA
1. **What:** Profiling dimensions, types, missingness, duplicates, distributions, trends, promotions, coverage and price-demand relationships.
2. **Why:** Every modelling choice came from an EDA finding: skew → log1p/WMAPE; promo-driven spikes → price/promo features; wrong-sign raw slopes → fixed effects.
3. **Why not jump to modelling:** I would have missed the duplicate-store join and the stockout issue, and trusted naive elasticities with the wrong sign.
4. **Assumptions:** Aggregates shown are representative; outliers are mostly real.
5. **What can go wrong:** Over-plotting and spurious patterns; reading causal meaning into scatter plots.
6. **Challenge:** "Name one EDA finding that changed your model."
7. **Concise:** "For private-label pretzels the raw price-units slope was positive, +1.8. That's what made me add store and week fixed effects; the slope became about −0.8."
8. **Deeper:** Stores that charge more for private label are busier stores, so the across-store comparison mixes store size with price. Demeaning within store removes that. Other examples: seasonal-naive being worst told me demand follows the promo calendar, not a stable yearly cycle; 8.7% missing grid weeks led to the NaN-not-zero decision.

## D. Feature Engineering
1. **What:** 43 features: weekly lags (1-4, 8, 13, 52), rolling means/std, expanding mean, availability count, target-week price/promo, promo lags, calendar/holiday/Super Bowl, store and product attributes.
2. **Why:** Lags capture level and persistence; price/promo capture planned decisions; promo lags capture post-promotion dips; calendar captures events.
3. **Why not raw time index / only lags:** Without price and promo the model can't explain promotion spikes, and can't answer what-if questions.
4. **Assumptions:** Target-week price and promotion are known in advance.
5. **What can go wrong:** Leakage via windows that include the current week; misaligned lags when weeks are missing.
6. **Challenge:** "Isn't using the current week's price leakage?"
7. **Concise:** "No. Leakage is information unavailable at prediction time. Prices and promotions are planned before the week starts. Same-week visits and spend are unknown in advance, so I excluded them."
8. **Deeper:** History features are built on a complete weekly calendar per store-product so `shift(1)` truly means "last week" even with gaps; notebook 02 asserts lag_1(t) = units(t−1) and that the rolling mean excludes week t. The *residual* risk is plan changes after forecasting. In production you'd snapshot the plan used for each forecast.

## E. Time Series
1. **What:** ~3,900 weekly series (store × product), gappy, ≤156 points, strongly promotion-driven.
2. **Why the global model:** One model across all series shares information (e.g. the promo effect learnt on Cheerios helps Froot Loops).
3. **Why not per-series ARIMA/Prophet:** Too short and gappy; exogenous promotions are handled poorly; 3,900 models to maintain.
4. **Assumptions:** Relationships are stable enough between 2009-10 and 2011.
5. **What can go wrong:** Structural breaks (the Quaker phase-out in late 2011), new products with no history.
6. **Challenge:** "Why is seasonal naive worse than naive?"
7. **Concise:** "Because demand here is driven by the promotion calendar, which changes year to year. Last week is more informative than the same week last year."
8. **Deeper:** Test WMAPE: seasonal naive 0.587, naive 0.433. Yearly patterns exist (Super Bowl for pretzels and pizza), but at store-product level most variance is promotional. This also explains why price and promotion features rank highly in SHAP.

## F. Forecasting
1. **What:** One-week-ahead forecast of units per store-product, trained on log1p(units), bias-corrected.
2. **Why:** Gives baseline demand Q0 at the current price for the optimizer.
3. **Why not multi-step:** The decision is weekly; the one-step setting matches its use. Multi-step would need recursive or direct strategies.
4. **Assumptions:** Planned prices/promos are correct inputs; the 2011 relationship resembles 2009-10.
5. **What can go wrong:** Retransformation bias (−5.5% before correction), drift, extrapolation on unseen prices.
6. **Challenge:** "Why did you correct bias, and how?"
7. **Concise:** "Because a log-target model predicts roughly the median, which under-forecasts total volume by 5.5%. I multiplied predictions by Σactual/Σpredicted from the last training fold (1.047). Bias fell to −1.0% with the same WMAPE."
8. **Deeper:** Jensen's inequality: E[exp(X)] ≥ exp(E[X]). Alternatives: Duan's smearing estimator, or a Poisson/Tweedie objective. I tried Poisson: nearly unbiased but WMAPE 0.292 vs 0.287. The factor is estimated on training data only, so the test stays clean. Results: XGBoost WMAPE 0.288, MAE 5.42, RMSE 11.29, a 34% WMAPE improvement over naive.

## G. XGBoost
1. **What:** Gradient-boosted trees: sequentially fit trees to the gradient of the loss, with shrinkage (learning rate), row/column subsampling and L2 regularisation. `hist` tree method; native categorical support.
2. **Why:** Best accuracy/effort for tabular data with interactions and missing values; fast; explainable with SHAP.
3. **Why not LSTM:** Short gappy series and tabular exogenous features; deep models would need more data and tuning for little expected gain, and are harder to explain.
4. **Assumptions:** Train and test come from a similar distribution; the features carry the signal.
5. **What can go wrong:** Overfitting (controlled with early stopping on walk-forward folds, min_child_weight=20, subsampling); **no extrapolation** beyond seen feature values.
6. **Challenge:** "How did you choose the number of trees?"
7. **Concise:** "Early stopping on each walk-forward fold, then the mean best iteration (268) when refitting on all training data."
8. **Deeper:** Params: lr 0.05, depth 8, min_child_weight 20, subsample 0.8, colsample 0.8, λ=1. I didn't do a big hyperparameter search: the gap to the baselines is large, and LightGBM with different defaults lands at the same WMAPE, so tuning isn't where the value is. Missing lags are routed by learned default directions. Categoricals use partition-based splits (`max_cat_to_onehot=1`).

## H. LightGBM
1. **What:** Gradient boosting with histogram binning and leaf-wise tree growth (grows the leaf with largest loss reduction).
2. **Why:** Robustness check. Same accuracy (0.287), so the result isn't an XGBoost artefact.
3. **Why not as primary:** Brief asked for XGBoost; they tie; one primary model keeps the system simple.
4. **Assumptions:** Same as XGBoost.
5. **What can go wrong:** Leaf-wise growth overfits small data without `num_leaves`/`min_child_samples` limits (127 / 40 here).
6. **Challenge:** "XGBoost vs LightGBM, the differences?"
7. **Concise:** "XGBoost grows trees level-wise by default; LightGBM grows leaf-wise, which is usually faster and deeper for the same number of leaves. Both use histograms. In practice accuracy is often equal, as here."
8. **Deeper:** LightGBM adds GOSS (gradient-based sampling) and EFB (exclusive feature bundling) for speed; native categorical handling via gradient-sorted splits. XGBoost has exact and hist methods, and strong monotone/interaction constraint support.

## I. Evaluation Metrics
1. **What:** MAE, RMSE, WMAPE (primary), MAPE, bias.
2. **Why WMAPE:** Volume-weighted, scale-free, robust to 1-unit weeks, and interpretable as "% of volume mis-forecast".
3. **Why not MAPE:** Dominated by tiny actuals (MAPE ≈0.55 vs WMAPE ≈0.29 for the same model); undefined at zero.
4. **Assumptions:** Errors on big and small series matter in proportion to volume.
5. **What can go wrong:** A low WMAPE can hide bias; RMSE is driven by big stores.
6. **Challenge:** "Why report bias separately?"
7. **Concise:** "Because a model can have good WMAPE and still systematically under-forecast. For revenue estimates, the total volume must be right."
8. **Deeper:** Bias = (Σŷ−Σy)/Σy. Our uncorrected log model was −5.5%; the corrected one −1.0%. For inventory you'd also look at quantile losses, because under- and over-stocking have different costs.

## J. Time-Series Validation
1. **What:** Final test = all of 2011 (untouched). Selection = 4 expanding walk-forward folds (13 weeks each) in 2009-10.
2. **Why:** Mimics deployment: train on the past, predict the future.
3. **Why not random K-fold:** Same and neighbouring weeks would be in train and validation, and rolling features overlap training targets. That is leakage and optimistic scores.
4. **Assumptions:** Future resembles the recent past.
5. **What can go wrong:** Early stopping on the validation fold slightly inflates CV scores (disclosed); a single test year may be unrepresentative.
6. **Challenge:** "Did you use sklearn's TimeSeriesSplit?"
7. **Concise:** "Same idea, but I split on week boundaries instead of row positions, so no week is split between train and validation."
8. **Deeper:** Fold WMAPEs for XGBoost: 0.335, 0.282, 0.284, 0.293. The first fold is worst because it has only one year of training and no lag-52 history. CV mean 0.298 vs test 0.288: consistent, no sign of test overfitting. A gap/embargo between train and validation isn't needed for one-step-ahead, because features only look back.

## K. Price Elasticity
1. **What:** % change in quantity per % change in price, estimated as the slope in a log-log regression.
2. **Why:** It's the "what-if" input the optimizer needs.
3. **Why not demand curves from XGBoost:** Its price response was monotone for only 43% of products; trees can't extrapolate and learn price mostly from promotions.
4. **Assumptions:** Constant elasticity near the current price; the identification strategy removes the main confounders; no cross-price effects; response to increases mirrors response to cuts.
5. **What can go wrong:** Confounding, reverse causality, selection, too little price variation, noisy small products.
6. **Challenge:** "How exactly did you use promotions to address confounding?"
7. **Concise:** "I identify the elasticity from temporary price reductions that had no ad or display, so the price cut isn't bundled with advertising. Store fixed effects remove store differences, and week fixed effects remove the chain's promotion timing. What's left is different stores discounting differently in the same week."
8. **Deeper:** The specification ladder shows the contamination: naive −2.07 → +promo controls −1.51 → +store & week FE −0.94; the promotion-identified M8 gives −0.84. I also tried a promotion-calendar instrument (other stores' discounts the same week). It was strong (F ≈ 1,900) but invalid: it reproduced the naive estimate because the chain schedules promotions into demand peaks. Noisy product estimates are shrunk toward the category mean (empirical Bayes), so every product keeps a product-level estimate. On 17,842 held-out 2011 price cuts, these elasticities have calibration slope ≈ 1.0, i.e. the right magnitude.

## L. Log-Log Regression
1. **What:** `log Q = α + β log P + controls + ε`; β = elasticity.
2. **Why:** β is directly the elasticity; multiplicative effects become additive; skew is compressed; and `Q(P)=Q0(P/P0)^β` is exactly the same model, so estimation and optimization are consistent.
3. **Why not linear Q~P:** Its elasticity changes along the curve and depends on scale; one slope is shared by products selling 2 or 2,000 units.
4. **Assumptions:** Constant elasticity; homoskedastic-ish errors on the log scale; Q > 0.
5. **What can go wrong:** Zero sales can't be logged (selection); constant elasticity breaks far from observed prices; exponentiating predictions has retransformation bias.
6. **Challenge:** "Why can't revenue maximization ever give an interior solution here?"
7. **Concise:** "Revenue = P·Q0(P/P0)^E ∝ P^(1+E), which is monotone in P. So the optimum is always at a bound: the lower one if E < −1, the upper one if E > −1."
8. **Deeper:** Profit has an interior optimum only if E < −1: P* = C·E/(1+E), i.e. margin −1/E. For −1 ≤ E < 0, profit also rises with P until a constraint stops it. So with constant elasticity, *constraints do much of the work*, which is why I report which constraint binds for each product.

## M. OLS
1. **What:** Minimise the sum of squared residuals; β̂ = (X'X)⁻¹X'y.
2. **Why:** Transparent, fast, interpretable, with standard errors and confidence intervals.
3. **Why not ML regressors for elasticity:** We need an interpretable, *controlled* partial effect with uncertainty, not predictive accuracy.
4. **Assumptions:** Linearity (in logs); **exogeneity E[ε|X]=0** (the critical one); no perfect multicollinearity; for SEs, the error structure must be handled. I cluster by store.
5. **What can go wrong:** Endogeneity biases β; serial correlation makes naive SEs too small; outliers have leverage.
6. **Challenge:** "Why cluster standard errors by store?"
7. **Concise:** "Weeks of the same store share unobserved shocks, so residuals are correlated within a store. Clustering by store accounts for that. Otherwise SEs are too small and CIs too narrow."
8. **Deeper:** Fixed effects are removed by demeaning (Frisch-Waugh-Lovell: regressing demeaned y on demeaned x gives the same slope as including all dummies). With two sets (store and week) I use alternating projections until convergence. Minor caveat: statsmodels doesn't know the absorbed FE degrees of freedom; with clustered SEs and ~77 clusters this matters little.

## N. Endogeneity
1. **What:** A regressor correlated with the error term, so OLS is biased and inconsistent.
2. **Why it matters here:** Managers set prices *knowing* things about demand (seasonal peaks, events, manufacturer-funded deals). Price is a choice, not an experiment.
3. **Why not ignore it:** It can flip signs (the private-label pretzel naive slope was +1.8).
4. **Assumptions in our fix:** Within a week, which stores run a deeper temporary discount is unrelated to store-specific demand shocks. This is untestable.
5. **What can go wrong:** A store discounting because of *local* demand (e.g. a local competitor's promotion).
6. **Challenge:** "Why not an instrumental variable?"
7. **Concise:** "I tried one: the same product's discount in other stores that week. It was very strong but invalid. It reproduced the naive estimate and failed out of sample, because chain-wide promotions are timed to demand peaks. Week fixed effects handle that timing better."
8. **Deeper:** IV needs relevance (testable: first-stage F) and exclusion (untestable, but checkable for plausibility). The red flags were that IV ≈ naive OLS, that IV ≫ fully controlled OLS for products whose variation is chain-wide timing (Frescetta), and that held-out MAE skill was −21.8%. A cost-shifter instrument (wholesale cost changes) would be better, but the data has no cost. A randomised price test is the gold standard.

## O. Confounding
1. **What:** A third variable driving both price and demand.
2. **Our confounders:** Feature ads and displays (come with discounts, raise demand by themselves), store size/clientele, week-level shocks (holidays, Super Bowl, national ads), chain-wide promotion timing.
3. **Why controls, not correlation:** The naive median −2.07 roughly halves to −0.94 with controls, so correlation vastly overstated price sensitivity.
4. **Assumptions:** Controls are measured well; no important unmeasured confounder remains.
5. **What can go wrong:** Over-controlling (week FE also remove *good* chain-wide price variation and inflate SEs: median SE 0.164 vs 0.069).
6. **Challenge:** "Why did adding feature/display make elasticities *less* negative?"
7. **Concise:** "Because feature and display coincide with discounts and lift sales by themselves. Without them, their lift is attributed to the lower price."
8. **Deeper:** Omitted-variable bias formula: bias = γ·δ, where γ is the effect of the omitted variable on demand (positive for ads) and δ is its association with log price (negative: ads come with lower prices). The product is negative, so the naive β is too negative. That's exactly what we see.

## P. Causal Inference
1. **What:** Estimating what *would* happen under an intervention, e.g. a price change.
2. **Why relevant:** Pricing is an intervention; the optimizer's output is only as good as the causal estimate.
3. **What we did:** Design-based identification from promotion events (TPR-only sample, store and week FE), a rejected IV, and out-of-sample validation on real price changes. All are clearly labelled as not fully causal.
4. **Assumptions:** No store-week-specific confounding; stable unit treatment (no spillovers between products or stores), which cannibalisation violates.
5. **What can go wrong:** Residual confounding, selection, interference, cuts ≠ increases.
6. **Challenge:** "Design the experiment that would validate your engine."
7. **Concise:** "Randomly assign matched stores to treatment and control, run the engine in treatment stores for 8-12 weeks, and estimate the effect with difference-in-differences on units, margin and basket size."
8. **Deeper:** Stratify by segment and size; pre-register metrics; size it with a power analysis from store-week variance; include category-level outcomes to catch cannibalisation and traffic; watch for spillovers; check legal/fairness rules. My held-out validation is a weaker, observational version of this: it shows the elasticity predicts real price-cut outcomes with the right magnitude.

## Q. Optimization and the dynamic engine
1. **What:** For each product × store segment: maximise (P − C)·Q0(P/P0)^E subject to 0.8P0 ≤ P ≤ 1.2P0, Q(P) ≥ 0.9Q0, P > C. Then move the shelf price toward the target by at most 5% per week, re-estimating elasticities every quarter.
2. **Why:** Makes the price/volume/margin trade-off explicit and auditable, and limits shocks to customers.
3. **Why not a rule ("+5%")?** It ignores cost and differences between products; in the replay it gives +15.4% simulated profit vs +34.3%.
4. **Assumptions:** Constant elasticity, ASSUMED cost, independent products, Q0 correct.
5. **What can go wrong:** Constraints, not data, set the size of the move; garbage elasticities give confident garbage prices.
6. **Challenge:** "What's dynamic about it? Does it actually beat a static change?"
7. **Concise:** "It recomputes targets weekly, re-estimates elasticities quarterly, prices store types differently and phases changes in at ≤5% a week. It does *not* beat a one-time static change in my simulation (+34.3% vs +35.0%). Its value is control and adaptivity, not extra profit."
8. **Deeper:** With constant elasticity the optimal markup doesn't depend on the demand level, so weekly targets only move at re-estimation. A truly demand-reactive engine would need elasticities that vary with demand state (holidays, promotions) or capacity constraints such as inventory. Guardrail price = P0·0.9^(1/E): +13% for E ≈ −0.8, +7% for E ≈ −1.6.

## R. SciPy
1. **What:** `scipy.optimize.minimize_scalar(method="bounded")`: Brent's method on an interval.
2. **Why:** 1-D, smooth, bounded: the textbook fit.
3. **Why not grid search/LP/gradient:** Grid is used as a cross-check (401 points); LP needs linear objectives; gradients are unnecessary for 1-D.
4. **Assumptions:** Unimodal objective on the interval (true for constant-elasticity profit).
5. **What can go wrong:** Bounded Brent never evaluates exactly at the bounds; the feasible interval can be empty (cost floor above the band for clearance items).
6. **Challenge:** "How do you know the optimizer is right?"
7. **Concise:** "Every recommendation is checked: optimizer equals grid argmax, matches the closed form clipped to bounds, constraints respected, price above cost, demand decreasing. All 141 product × segment recommendations pass."
8. **Deeper:** I compare the objective at both bounds with the Brent solution and snap to a bound if it's better. If the cost floor lies above the band, there's no profitable feasible price; the engine charges the highest allowed price and flags it. I found this case during the 20%-margin stress test, when the optimizer crashed with "lower bound exceeds upper bound".

## S. Historical backtesting and counterfactual simulation
1. **What:** Two backtests, both estimated on 2009-10 only. (1) **Price-response validation:** predict the lift of 17,842 real 2011 store-weeks with a pure price cut and compare with what happened. (2) **Walk-forward replay** of the dynamic engine over 2011 (quarterly re-estimation, weekly paths), with outcomes simulated by units × m^E.
2. **Why:** (1) checks the model against reality; (2) evaluates the *decision policy* out of sample.
3. **Why not only (2)?** (2) is circular: a policy judged by the elasticity that built it looks good by construction. (1) is not.
4. **Assumptions:** (1) non-promoting stores the same week are a fair comparison; (2) elasticity correct and stable, competitors and promotions unchanged.
5. **What can go wrong:** (1) covers price cuts only; (2) depends on the assumed cost and elasticity.
6. **Challenge:** "How do you know your elasticities are right?"
7. **Concise:** "On held-out 2011 price cuts, predicted and observed lifts line up with calibration slope ≈ 1.0. The naive OLS overstates the response 2.5×, and the IV is worse than ignoring price."
8. **Deeper:** MAE skill vs a zero-effect model: engine 6.0%, controlled OLS 7.8%, naive 1.7%, IV −21.8%. Skill is modest because store-weeks are noisy; calibration is what matters for pricing. Replay: +34.3% simulated assumed-cost profit; +27.9% to +39.6% under alternative elasticities including a different model; +21% to +70% depending on the assumed margin. These are model-based estimates, not observed outcomes.

## T. SHAP
1. **What:** Shapley-value attributions: each feature's additive contribution to a prediction relative to the average.
2. **Why:** Global importance + local explanations; exact for trees (TreeExplainer).
3. **Why not gain importance:** Biased toward high-cardinality/frequently-split features; no direction or local view.
4. **Assumptions:** Interventional/tree-path assumptions about feature independence; correlated features share credit arbitrarily.
5. **What can go wrong:** Reading SHAP as causal; correlated lags splitting importance.
6. **Challenge:** "SHAP says price matters. Can you use that as elasticity?"
7. **Concise:** "No. SHAP explains how the model uses price, which is an associational pattern learned mostly from promotions. The optimizer uses the controlled elasticity for the price effect."
8. **Deeper:** Top drivers (mean |SHAP|, log scale): expanding mean 0.335, lag 1 0.211, UPC 0.097, feature ad 0.068, rolling mean 4 0.066, price vs recent base 0.061. Demand history sets the level; promotion flags and relative price explain deviations. Values are on the log1p scale, so they're multiplicative effects on units.

## U. Limitations
1. **What:** Observational elasticity; validated on price cuts but applied to increases; assumed cost; missing zero-sales weeks; constant elasticity; single-product optimisation; simulated (not observed) policy gains; old single-retailer data.
2. **Why state them:** Pricing errors cost real money; stakeholders must know which numbers are measured vs assumed.
3. **Most important:** The cost assumption (it scales the headline gain from +21% to +70%) and the cut-vs-increase extrapolation.
4. **Mitigations used:** Promotion-based identification, held-out validation, stress tests, margin sensitivity, guardrails, gradual paths, sanity checks, distribution checks.
5. **What can still go wrong:** Real traffic loss exceeds what the guardrail allows for; competitors undercut.
6. **Challenge:** "What would you do with one more month?"
7. **Concise:** "Cross-price elasticities within categories, a full hierarchical Bayesian model, and then design a price test."
8. **Deeper:** Also demand-state-dependent elasticities (so weekly prices react to forecasted peaks), real cost data, and validating price *increases* using the few regular-price increases in the data (difference-in-differences across stores).

## V. Productionization
1. **What:** Weekly batch: ingest → validate → features → forecast Q0 → refresh elasticities (monthly) → optimise → review → publish price file; plus monitoring.
2. **Why batch:** Regular prices change weekly at most; no real-time need.
3. **Why not full automation:** Price changes need human approval and legal checks; the model ignores competitors.
4. **Assumptions:** Upstream data arrives on time and in schema; the plan for next week is known.
5. **What can go wrong:** Data drift, schema changes, new products (cold start), forecast bias drift, elasticity instability.
6. **Challenge:** "How would you monitor this?"
7. **Concise:** "Track forecast WMAPE and bias weekly by category, elasticity stability over refits, share of recommendations at constraints, and realised vs expected units after price changes."
8. **Deeper:** Version data/models/config (MLflow or similar); keep `category_levels.json`-style artifacts so scoring matches training (that bug surfaced during this project); unit tests for leakage (lags only look back) and optimizer checks; a holdout of stores for continuous experimentation; alerting when bias exceeds ±3%; retrain forecasts weekly and elasticities monthly with shrinkage.

---

## Rapid-fire questions
- **Why log1p and not log?** Forecast targets could be 0 in principle; log1p is safe. Elasticity uses log because only Q>0 rows enter.
- **What's a calibration slope?** Regress observed lift on predicted lift. 1 = right magnitude, <1 = model overstates the response.
- **What's the difference between a strong and a valid instrument?** Strong = it moves the endogenous variable (testable). Valid = it affects the outcome only through that variable (untestable; mine failed).
- **What does shrinkage do to a precise estimate?** Almost nothing: the weight on its own estimate is τ²/(τ²+se²) ≈ 1 when se is small.
- **Why is P0 demand-weighted?** Then segment revenue under a uniform multiplier equals P0·Q0·m^(1+E) exactly, so the 1-D optimization is exact.
- **Why 5% per week?** Avoids sticker shock and lets the engine react to re-estimated elasticities before a full move; it costs a little early profit (+34.3% vs +35.0% static).
- **What did you get wrong along the way?** A naive store join duplicated 13,693 rows; categorical codes silently changed after a merge (fixed by persisting category levels); an IV that looked great (F ≈ 1,900) turned out invalid; an exploratory backtest figure wasn't reproduced by the final pipeline, so I removed it.
