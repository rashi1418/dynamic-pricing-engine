"""Project-wide constants.

Everything that is an *assumption* (not observed in the data) is collected here
and labelled ASSUMED so it can never be confused with observed data.
"""
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports" / "results"

RAW_FILE = DATA_RAW / "dunnhumby_breakfast_at_the_frat.xlsx"
PANEL_FILE = DATA_PROCESSED / "panel_clean.parquet"
FEATURES_FILE = DATA_PROCESSED / "features.parquet"

# --------------------------------------------------------------------------
# Data source (a byte-identical public mirror of the dunnhumby Source File).
# The original is distributed by dunnhumby at https://www.dunnhumby.com/source-files/
# The mirror is pinned to a commit so the download is reproducible.
# --------------------------------------------------------------------------
RAW_URL = (
    "https://raw.githubusercontent.com/Feanor-22/Breakfast-at-the-Frat/"
    "7c58641883c787c245834bc40e89aab48da80daf/Breakfast_at_the_Frat.xlsx"
)
RAW_MD5 = "275baff94d683bba142d06c32470cf2c"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Time splits (weekly data, weeks end on Wednesdays)
#   2009-01-14 ... 2010-12-29  -> TRAIN  (103 weeks)
#   2011-01-05 ... 2012-01-04  -> TEST   ( 53 weeks, never touched while fitting)
# --------------------------------------------------------------------------
TRAIN_END = pd.Timestamp("2010-12-29")
TEST_START = pd.Timestamp("2011-01-05")
N_CV_FOLDS = 4          # walk-forward folds inside the training period
CV_VAL_WEEKS = 13       # each validation fold = one quarter

# --------------------------------------------------------------------------
# Elasticity estimation: minimum data requirements per product
# --------------------------------------------------------------------------
MIN_OBS = 500                 # store-weeks with positive sales
MIN_DISTINCT_PRICES = 10      # distinct shelf prices
MIN_WITHIN_LOG_PRICE_SD = 0.03  # sd of log price after removing store-product means (~3%)

# --------------------------------------------------------------------------
# Optimization assumptions  (ASSUMED - the dataset contains NO cost data)
# --------------------------------------------------------------------------
ASSUMED_GROSS_MARGIN = 0.30   # ASSUMED: unit cost = (1 - 0.30) x reference regular price
MAX_PRICE_CHANGE = 0.20       # business rule: new price within +/-20% of current price
MAX_VOLUME_LOSS = 0.10       # business rule: expected units may not fall more than 10%
                              # (protects store traffic / market share, which the
                              #  single-product profit objective ignores)
REFERENCE_WEEKS = 13
MIN_CURRENT_DISTRIBUTION = 0.25
# Dynamic engine
MAX_WEEKLY_CHANGE = 0.05      # business rule: price moves at most 5% per week (gradual transitions)
REESTIMATE_EVERY_WEEKS = 13   # elasticities re-estimated every quarter on all data available so far  # only price products still sold in >= 25% of their usual stores          # "current price" = median regular price over last 13 weeks

# US Super Bowl dates (public facts) - relevant for pretzels / frozen pizza
SUPER_BOWL_DATES = pd.to_datetime(["2009-02-01", "2010-02-07", "2011-02-06"])
