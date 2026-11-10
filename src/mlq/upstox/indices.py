"""Download market context: broad indices, sector indices, India VIX, 10y govt bond index.

Why: a stock's move only means something relative to the market/sector, and VIX tells us
how nervous the market is (regime). Same throttled, resumable downloader as stocks.
Saved to data/raw/upstox_idx/{1d,5m}/<name>.parquet (file name = index name, spaces -> _).

uv run python -m mlq.upstox.indices
"""
import sys
from datetime import date

import polars as pl

from mlq.config import DATA, INSTRUMENTS_FILE
from mlq.upstox.auth import load_token
from mlq.upstox.history import Throttle, download

ROOT = DATA / "raw" / "upstox_idx"

BROAD = ["Nifty 50", "Nifty Next 50", "Nifty 100", "Nifty 200", "Nifty 500",
         "NIFTY MIDCAP 100", "NIFTY SMLCAP 100", "India VIX"]
SECTORS = ["Nifty Bank", "Nifty Pvt Bank", "Nifty PSU Bank", "Nifty Fin Service", "Nifty IT",
           "Nifty Auto", "Nifty Pharma", "NIFTY HEALTHCARE", "Nifty FMCG", "Nifty Metal",
           "Nifty Energy", "NIFTY OIL AND GAS", "Nifty Realty", "Nifty Infra", "Nifty Media",
           "Nifty PSE", "Nifty CPSE", "NIFTY CONSR DURBL", "Nifty Commodities",
           "Nifty Consumption"]
RATES = ["Nifty GS 10Yr"]
JOBS = [("days", 1, date(2005, 1, 1)), ("minutes", 5, date(2022, 1, 3))]


def fname(name: str) -> str:
    return name.replace(" ", "_")


def main():
    idx = pl.read_parquet(INSTRUMENTS_FILE).filter(pl.col("segment") == "NSE_INDEX")
    keys = dict(idx.select(["name", "instrument_key"]).iter_rows())
    wanted = BROAD + SECTORS + RATES
    missing = [n for n in wanted if n not in keys]
    if missing:
        print("not in instrument list:", missing, flush=True)
    token, th, end = load_token(), Throttle(1.0), date.today()
    failed, streak = [], 0
    for unit, interval, start in JOBS:
        names = wanted if unit == "days" else [n for n in wanted if n not in RATES]
        for i, n in enumerate([n for n in names if n in keys], 1):
            try:
                download(fname(n), keys[n], unit, interval, start, end, token, th, root=ROOT)
                print(f"[{unit}{interval}] {i} {n}", flush=True)
                streak = 0
            except Exception as e:
                print(f"[{unit}{interval}] FAIL {n}: {e}", flush=True)
                failed.append((unit, n))
                streak += 1
                if "401" in str(e):
                    sys.exit("Token rejected (401).")
                if streak >= 3:
                    sys.exit("3 failures in a row (network?). Rerun to resume.")
    print("DONE. failed:", failed, flush=True)


if __name__ == "__main__":
    main()
