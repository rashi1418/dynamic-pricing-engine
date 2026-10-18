# Dataset Selection

## What the project needs

This is a **pricing** project, not a forecasting competition. A dataset is only useful if it lets us
estimate how demand responds to price. The requirements, in priority order:

1. **Observed selling price and observed units at the same grain** (product x store x time). Non-negotiable.
2. **Repeated price observations for the same product**, with real variation (otherwise elasticity is not identifiable).
3. **Promotion information** that can be separated from price. Promotions are the main confounder.
4. Store and product attributes, for fixed effects and features.
5. Enough history for time-series validation (at least 2 years, so yearly seasonality and a clean holdout exist).
6. Real, not synthetic, data, and a **reproducible, legitimate download path**.

## Candidates considered

| # | Dataset | Source | Key columns | Strengths | Weaknesses | Verdict |
|---|---|---|---|---|---|---|
| 1 | **dunnhumby "Breakfast at the Frat"** | dunnhumby Source Files (https://www.dunnhumby.com/source-files/) | week, store, UPC, units, visits, households, spend, **price, base price**, feature, display, TPR flag; store & product lookups | Price **and** regular (base) price; promotion type flags; 156 weeks; 77 stores; 55 products in 4 categories; designed by dunnhumby for price-sensitivity analysis | Only 55 products; no cost; no inventory; weeks without sales have no row | **Selected** |
| 2 | Dominick's Finer Foods (Kilts Center, Chicago Booth) | https://www.chicagobooth.edu/research/kilts/datasets/dominicks | week, store, UPC, units (MOVE), price, bundle qty, profit margin, deal code | Huge (store-week-UPC, ~9 years, 3,500+ UPCs, 29 categories); even contains a gross-margin field | Data from 1989-1997; one file pair per category in SAS/CSV; the host was **not reachable from the build environment**, so the pipeline could not be tested end to end | Rejected (practical); best alternative for a v2 |
| 3 | dunnhumby "The Complete Journey" | dunnhumby Source Files | household transactions, sales value, retail/coupon discount | Rich household panel, 2 years | Transaction-level; shelf price must be reverse-engineered from sales value / quantity and discounts; better suited to customer analytics | Rejected |
| 4 | M5 Forecasting (Walmart) | Kaggle | daily units per item-store, weekly `sell_prices`, calendar, SNAP events | Large, famous, has prices | No promotion flags; prices at weekly grain vs daily units; Kaggle login required; explicitly asked not to default to famous Walmart data | Rejected |
| 5 | Corporación Favorita | Kaggle | daily unit sales, `onpromotion` | Big, promo flag | **No price column**, so no elasticity is possible | Rejected |
| 6 | Rossmann Store Sales | Kaggle | store-level daily sales, promo | Classic forecasting set | Store-level revenue only: **no product, no price** | Rejected |
| 7 | Walmart Store Sales (2014 recruiting) | Kaggle | weekly department sales, markdowns | Holidays, markdown columns | **No product price or units** | Rejected |
| 8 | UCI Online Retail (I/II) | UCI ML Repository | invoice, stock code, quantity, unit price, customer | Real prices and quantities | Wholesale-like e-commerce; many one-off items; no promotion data; price variation is mostly per-customer quantity pricing, not retail price changes | Rejected |

## Why "Breakfast at the Frat"

- It contains **both** the actual shelf price (`PRICE`) and the regular price (`BASE_PRICE`). This lets us separate
  *temporary-discount* variation (promotions) from regular-price variation, and identify the elasticity from
  temporary price reductions. This is the backbone of notebook 03.
- Promotions are recorded as three separate mechanisms: `FEATURE` (circular ad), `DISPLAY` (in-store display) and
  `TPR_ONLY` (temporary price reduction only). Feature and display are the classic confounders of a
  price-demand regression. Because they are recorded separately, pure price cuts can be isolated and used to identify the elasticity.
- Stores carry a segment label (value / mainstream / upscale), which the dynamic engine uses to price store types differently.
- The price variation is real and substantial. Measured within the same store and product, the standard deviation of
  log price is 6-27%. All 55 products pass the minimum-data rules (`reports/results/price_variation.csv`).
- A panel of 77 stores x 156 weeks supports store and week fixed effects, and a full calendar year (2011) as a holdout.
- dunnhumby publishes it specifically for "price sensitivity analysis" and "promotional effectiveness", which matches the
  project's purpose.

## Exact dataset used

- File: `Breakfast_at_the_Frat.xlsx` (34,051,636 bytes, MD5 `275baff94d683bba142d06c32470cf2c`)
- Period: weeks ending 2009-01-14 to 2012-01-04 (156 weeks)
- 524,950 transaction rows, 77 stores, 55 products
- Categories: bag snacks (pretzels), cold cereal, frozen pizza, oral hygiene (mouthwash)

## Acquisition and licence notes

- The dataset is distributed free of charge by dunnhumby through its Source Files programme, for educational and
  research use, subject to dunnhumby's terms. Direct download requires a web form, so it cannot be automated.
- For reproducibility the pipeline downloads the workbook from a public GitHub mirror **pinned to a commit hash** and
  verifies the MD5 checksum. The identical checksum was found on a second, independent mirror.
- The raw data is **not** redistributed in this repository (`.gitignore`). Manual download instructions are in
  `data/README.md`.
- No data was synthesized. The only non-observed quantity is the **assumed unit cost** (30% gross margin), which is
  labelled as an assumption wherever it is used.
