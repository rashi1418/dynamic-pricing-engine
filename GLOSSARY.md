# Glossary: Dynamic Pricing Engine for Retail Demand Optimization
(simple explanations, with how each term shows up in this project)

**Demand.** How many units customers want to buy at a given price and time. We only observe *sales*, which equal demand only if the product was in stock.

**Revenue.** Price × units sold. `R(P) = P · Q(P)`.

**Profit.** (Price − unit cost) × units. `Π(P) = (P − C) · Q(P)`. In this project C is **assumed** (30% gross margin), because the data has no cost.

**Price elasticity of demand.** % change in quantity ÷ % change in price. E = −1.5 means a 1% price rise is associated with ≈1.5% fewer units.

**Elastic demand.** |E| > 1: quantity reacts more than proportionally, so raising price *lowers* revenue (e.g. DiGiorno Three Meat pizza, about −2.1).

**Inelastic demand.** |E| < 1: quantity reacts less than proportionally, so raising price *raises* revenue (e.g. Honey Nut Cheerios 12.25 oz, about −0.1).

**Time series.** Observations ordered in time, where order matters (this week depends on last week). Here: ~3,900 weekly store × product series.

**Lag feature.** A past value used as input, e.g. `units_lag_1` = units last week. Lag 52 = same week last year.

**Rolling feature.** A statistic over a moving past window, e.g. mean of the previous 4 weeks. It must end *before* the week being predicted.

**Data leakage.** Using information at training time that would not be available at prediction time, which makes models look better than they are. Examples avoided here: same-week `visits`/`spend`, rolling windows that include the current week, random splits of time series.

**TimeSeriesSplit / walk-forward validation.** Train on the past, validate on the next block, then grow the training window and repeat. Here: 4 expanding folds of 13 weeks each inside 2009-2010.

**XGBoost.** Gradient-boosted decision trees. Trees are added one at a time, each fixing the previous ensemble's errors, with regularisation. Our main forecaster.

**LightGBM.** Another gradient-boosting library. It grows trees leaf-wise and bins features into histograms, which is fast. Used here as a cross-check (same accuracy).

**SHAP.** A method that splits a single prediction into additive contributions from each feature (based on Shapley values from game theory). It explains the model, not cause and effect.

**OLS (ordinary least squares).** Linear regression that picks coefficients minimising the sum of squared residuals. Our elasticity models are OLS on logs.

**Log-log model.** `log Q = α + β log P + ε`. β is the elasticity, assumed constant.

**Fixed effects.** A separate intercept per store (or week, or product). They remove everything constant within that group. Implemented by demeaning (the within transformation).

**Confounding.** A third factor that affects both price and demand and creates a misleading correlation. Example: feature ads come with price cuts *and* raise demand on their own.

**Endogeneity.** When an explanatory variable (price) is correlated with the error term, e.g. because managers set prices in response to expected demand. OLS is then biased.

**Reverse causality.** Demand influencing price, instead of price influencing demand (e.g. raising prices when demand is strong).

**Selection bias.** The sample is not representative. Here, weeks with zero sales are missing, so very-high-price weeks with no sales never enter the regression.

**Causal inference.** Methods for estimating what *would* happen under an intervention (e.g. a price change), not just what is correlated. The gold standard is a randomised experiment.

**Counterfactual.** The outcome that would have occurred under a different decision, which by definition is never observed. "What would 2011 sales have been at 8% higher prices?"

**Optimization.** Choosing the value of a decision variable (price) that maximises or minimises an objective, subject to constraints.

**Objective function.** The quantity being optimised: here profit `(P − C)·Q(P)` (or revenue).

**Constraint.** A rule the solution must satisfy: price within ±20% of current, expected units at least 90% of current, price above cost.

**Feasible region.** All prices that satisfy every constraint. The optimizer searches only inside it.

**Binding constraint.** A constraint the solution sits exactly on. For 94 of 141 product × store-segment recommendations the volume guardrail binds, meaning the optimizer would go further if allowed.

**Lerner condition.** At the profit-maximising price, margin (P − C)/P = −1/E. Used here to show that the estimated elasticities imply higher margins than retailers typically choose.

**Backtesting.** Evaluating a strategy on a past period it was not built on. Here it is an *offline counterfactual simulation* on 2011.

**WMAPE.** Σ|actual − forecast| ÷ Σ actual. A volume-weighted percentage error, robust to small actuals.

**Retransformation bias.** Training on log(y) and exponentiating the prediction gives a value near the median, which is below the mean for skewed data. That causes systematic under-forecasting (−5.5% here, corrected to −1.0%).

**Dynamic pricing.** Prices that are recomputed as conditions and data change. Here: targets recomputed weekly per product and store type,
elasticities re-estimated every quarter, prices moved toward the target by at most 5% per week. (Not real-time pricing.)

**Temporary price reduction (TPR).** A short-term price cut. In this data `TPR_ONLY = 1` means a price cut without feature ad or display.
The decision elasticity is identified from these events.

**Promotion-based identification.** Using promotion events as the source of price variation while designing away their confounders: only
pure price cuts (no ad/display), store and week fixed effects, so the estimate comes from stores discounting differently in the same week.

**Instrumental variable (IV) / 2SLS.** A variable that shifts price but affects demand *only through* price. 2SLS first predicts price from the
instrument, then regresses demand on that prediction. Tried here with the chain's promotion calendar; rejected.

**First-stage F-statistic.** Measures instrument strength; F < 10 is the usual warning sign of a weak instrument. Here median ≈ 1,900 (strong).

**Exclusion restriction.** The untestable IV assumption that the instrument has no direct path to the outcome. It failed here: the chain
schedules promotions into high-demand weeks.

**Empirical-Bayes shrinkage (partial pooling).** Pulling each noisy estimate toward its group mean by a weight τ²/(τ²+se²): precise estimates
stay put, noisy ones move toward the category mean. Gives every product a usable product-level elasticity.

**Calibration slope.** Slope of observed outcomes regressed on predictions. 1 = predictions have the right size; < 1 = the model overstates effects.
The engine's elasticities have ≈ 1.0 on held-out 2011 price cuts; the naive OLS has 0.39.

**Walk-forward backtest.** Re-running a policy through a past period using, at each decision date, only the data available at that time.

**Store segment.** dunnhumby's store classification: VALUE, MAINSTREAM or UPSCALE. The engine prices each segment separately.
