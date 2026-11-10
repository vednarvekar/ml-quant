"""Bulk download for a symbol list. Shared throttle keeps us under Upstox limits
(2000 req / 30 min; we use <= 1 req/s). Resumable: rerun to fill gaps / extend.

uv run python -m mlq.upstox.bulk            # daily + 5m for Nifty 100
"""
import sys
import time
from datetime import date

import polars as pl

from mlq.config import DATA
from mlq.upstox.auth import load_token
from mlq.upstox.history import Throttle, download
from mlq.upstox.instruments import equities

JOBS = [("days", 1, date(2005, 1, 1)), ("minutes", 5, date(2022, 1, 3))]


def main(list_csv=DATA / "nifty100_list.csv"):
    wl = pl.read_csv(list_csv)
    eq = equities()
    m = wl.join(eq.select(["isin", "instrument_key", "trading_symbol"]),
                left_on="ISIN Code", right_on="isin", how="left")
    missing = m.filter(pl.col("instrument_key").is_null())["Symbol"].to_list()
    if missing:
        print("No Upstox key for:", missing, flush=True)
    m = m.filter(pl.col("instrument_key").is_not_null())
    token, th, end = load_token(), Throttle(1.0), date.today()
    failed, streak = [], 0
    for unit, interval, start in JOBS:
        for i, row in enumerate(m.iter_rows(named=True), 1):
            sym = row["Symbol"]
            t0 = time.time()
            try:
                download(sym, row["instrument_key"], unit, interval, start, end, token, th)
                print(f"[{unit}{interval}] {i}/{m.height} {sym} {time.time()-t0:.0f}s", flush=True)
            except Exception as e:  # keep going; report at the end
                print(f"[{unit}{interval}] FAIL {sym}: {e}", flush=True)
                failed.append((unit, interval, sym))
                streak += 1
                if "401" in str(e):
                    sys.exit("Token rejected (401). Stopping.")
                if streak >= 3:
                    sys.exit("3 failures in a row (network down?). Stopping; rerun to resume.")
            else:
                streak = 0
    print("DONE. failed:", failed, flush=True)


if __name__ == "__main__":
    main()
