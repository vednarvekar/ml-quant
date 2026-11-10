"""Download the extra (non-current-Nifty-100) stocks that were in the point-in-time top-100
universe since 2022 and are still listed, so backtests aren't survivorship-biased.

Run ONLY when no other Upstox download is running (shared rate limit).
Saves into the same data/raw/upstox/{1d,5m}/ folders as the main universe.

uv run python -m mlq.upstox.bulk_universe
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


def targets(since: str = "2022-01-01", top_n: int = 100) -> pl.DataFrame:
    u = pl.read_parquet(DATA / "clean" / "universe.parquet")
    cur = set(pl.read_csv(DATA / "nifty100_list.csv")["Symbol"].to_list())
    names = (u.filter((pl.col("month") >= pl.lit(since).str.to_date()) & (pl.col("rank") <= top_n))
             ["name"].unique().to_list())
    names = [n for n in names if n not in cur]
    eq = equities().select(["trading_symbol", "instrument_key"])
    return eq.filter(pl.col("trading_symbol").is_in(names)).sort("trading_symbol")


def main():
    t = targets()
    print(f"{t.height} extra symbols", flush=True)
    token, th, end = load_token(), Throttle(1.0), date.today()
    failed, streak = [], 0
    for unit, interval, start in JOBS:
        for i, (sym, key) in enumerate(t.iter_rows(), 1):
            t0 = time.time()
            try:
                download(sym, key, unit, interval, start, end, token, th)
                print(f"[{unit}{interval}] {i}/{t.height} {sym} {time.time()-t0:.0f}s", flush=True)
                streak = 0
            except Exception as e:
                print(f"[{unit}{interval}] FAIL {sym}: {e}", flush=True)
                failed.append((unit, sym))
                streak += 1
                if "401" in str(e):
                    sys.exit("Token rejected (401).")
                if streak >= 3:
                    sys.exit("3 failures in a row (network?). Rerun to resume.")
    print("DONE. failed:", failed, flush=True)


if __name__ == "__main__":
    main()
