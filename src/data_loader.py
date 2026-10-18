"""Download and load the dunnhumby 'Breakfast at the Frat' workbook.

The workbook has three data sheets:
    dh Transaction Data  - one row per (week, store, UPC) with units, price, promo flags
    dh Store Lookup      - store attributes
    dh Products Lookup   - product attributes
Row 0 of every sheet is a title banner, so the header is on row 1.
"""
import hashlib
import urllib.request

import pandas as pd

from . import config


def _md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(force: bool = False) -> None:
    """Download the workbook from the pinned mirror and verify its checksum."""
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if config.RAW_FILE.exists() and not force:
        print(f"Raw file already present: {config.RAW_FILE}")
    else:
        print(f"Downloading {config.RAW_URL}")
        urllib.request.urlretrieve(config.RAW_URL, config.RAW_FILE)
    digest = _md5(config.RAW_FILE)
    if digest != config.RAW_MD5:
        raise ValueError(
            f"Checksum mismatch ({digest} != {config.RAW_MD5}). The file is not the "
            "expected dunnhumby workbook. Download it manually - see data/README.md."
        )
    print("Checksum OK:", digest)


def load_raw():
    """Return (transactions, stores, products) exactly as stored in the workbook.

    Reading the xlsx takes ~1 minute, so a parquet cache is written the first time.
    """
    cache = {n: config.DATA_PROCESSED / f"raw_{n}.parquet" for n in ("tx", "stores", "products")}
    if all(p.exists() for p in cache.values()):
        return tuple(pd.read_parquet(cache[n]) for n in ("tx", "stores", "products"))

    if not config.RAW_FILE.exists():
        raise FileNotFoundError(
            f"{config.RAW_FILE} not found. Run `python run_pipeline.py --download` "
            "or follow data/README.md."
        )
    sheets = {"tx": "dh Transaction Data", "stores": "dh Store Lookup", "products": "dh Products Lookup"}
    out = {}
    for name, sheet in sheets.items():
        df = pd.read_excel(config.RAW_FILE, sheet_name=sheet, header=1)
        df.columns = [str(c).strip().lower() for c in df.columns]
        out[name] = df
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_parquet(cache[name], index=False)
    return out["tx"], out["stores"], out["products"]
