# Data

## Source
**dunnhumby - "Breakfast at the Frat: A Time Series Analysis"** (dunnhumby Source Files programme).
Original distribution: https://www.dunnhumby.com/source-files/ (free download, registration form).

The file is **not committed** to this repository. Respect dunnhumby's terms of use.

## Automated acquisition (default)
```bash
python run_pipeline.py --download
```
downloads `Breakfast_at_the_Frat.xlsx` from a public GitHub mirror pinned to a commit:

```
https://raw.githubusercontent.com/Feanor-22/Breakfast-at-the-Frat/7c58641883c787c245834bc40e89aab48da80daf/Breakfast_at_the_Frat.xlsx
```
and verifies its MD5 checksum `275baff94d683bba142d06c32470cf2c`. The same checksum was
confirmed on a second, independent mirror (`mehwish2525/dunnhumby-_Breakfast-at-the-Frat`),
which is strong evidence the file is the unmodified original workbook.

## Manual acquisition (if the mirror disappears)
1. Download "Breakfast at the Frat" from https://www.dunnhumby.com/source-files/
2. Save the workbook as `data/raw/dunnhumby_breakfast_at_the_frat.xlsx`
3. Run `python run_pipeline.py` (without `--download`). If the checksum of your copy differs,
   the file may be a newer release; the pipeline still runs, but the numbers in the README may change.

## Workbook contents
| Sheet | Rows | Content |
|---|---|---|
| `dh Transaction Data` | 524,950 | one row per week x store x UPC |
| `dh Store Lookup` | 79 | store attributes (2 stores listed twice - see cleaning log) |
| `dh Products Lookup` | 58 | product attributes (55 appear in transactions) |
| `Glossary` | - | column definitions |

Transaction columns: `WEEK_END_DATE, STORE_NUM, UPC, UNITS, VISITS, HHS, SPEND, PRICE,
BASE_PRICE, FEATURE, DISPLAY, TPR_ONLY`.

## What is NOT in the data
- **Product cost / margin.** Profit figures in this project use an ASSUMED 30% gross margin
  (`src/config.py`). They are not observed.
- **Inventory / stockouts.** A store-week with no sales has no row. We cannot tell zero demand from
  out-of-stock or not-ranged.
- **Competitor prices.**

## Generated files (`data/processed/`, created by the pipeline)
- `raw_*.parquet`: fast cache of the three sheets
- `panel_clean.parquet`: cleaned store x product x week panel
- `features.parquet`: complete weekly grid + forecasting features
- `next_week_features.parquet`: feature rows for the first unobserved week (used by the app)
