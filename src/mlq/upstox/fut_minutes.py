"""1-minute candles for LIVE stock-futures contracts (near + next month), saved every evening.

Why: the strategy trades stock futures, but backtest and paper fills used the cash-market open.
Upstox serves minute history only while a contract is alive (expired contracts need a paid plan),
so we archive it daily. Nobody has to watch the market: this runs after the close.

Files: data/raw/upstox_fut/1m/{SYMBOL}_{EXPIRY}.parquet   (ts, open, high, low, close, volume, oi)

uv run python -m mlq.upstox.fut_minutes            # all F&O stocks, near + next contract
"""
import sys
from datetime import date, datetime, timedelta

import polars as pl

from mlq.config import DATA, INSTRUMENTS_FILE
from mlq.upstox.auth import load_token
from mlq.upstox.history import Throttle, download

ROOT = DATA / "raw" / "upstox_fut"
N_CONTRACTS = 2          # near + next month
LOOKBACK_DAYS = 100      # a contract lists about 3 months before expiry


def contracts(n: int = N_CONTRACTS) -> pl.DataFrame:
    """Live stock-futures contracts: sym, key, expiry (date), lot_size; first n expiries per stock."""
    d = pl.read_parquet(INSTRUMENTS_FILE).filter((pl.col("segment") == "NSE_FO")
                                                 & (pl.col("instrument_type") == "FUT")
                                                 & (pl.col("underlying_type") == "EQUITY"))
    d = d.with_columns(pl.col("expiry").map_elements(lambda x: datetime.fromtimestamp(x / 1000).date(),
                                                     return_dtype=pl.Date).alias("expiry"))
    d = d.sort(["underlying_symbol", "expiry"]).with_columns(
        pl.col("expiry").cum_count().over("underlying_symbol").alias("n"))
    return d.filter(pl.col("n") <= n).select([pl.col("underlying_symbol").alias("sym"),
                                              pl.col("instrument_key").alias("key"), "expiry", "lot_size"])


def file_stem(sym: str, expiry: date) -> str:
    return f"{sym}_{expiry.isoformat()}"


def refresh(min_interval: float = 0.5) -> dict:
    """Fetch what is missing for every live contract. ~400 light requests a day, resumable."""
    c = contracts()
    token, th, end = load_token(), Throttle(min_interval), date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    failed, streak = [], 0
    for sym, key, expiry, _ in c.iter_rows():
        try:
            download(file_stem(sym, expiry), key, "minutes", 1, start, end, token, th, root=ROOT)
            streak = 0
        except Exception as e:                                   # noqa: BLE001
            failed.append((sym, str(e)[:80]))
            streak += 1
            if streak >= 5:
                break                                            # network/token trouble: try again tomorrow
    return {"contracts": c.height, "failed": failed}


def load(sym: str, expiry: date) -> pl.DataFrame | None:
    f = ROOT / "1m" / f"{file_stem(sym, expiry)}.parquet"
    return pl.read_parquet(f) if f.exists() else None


if __name__ == "__main__":
    r = refresh()
    print(r)
    sys.exit(1 if len(r["failed"]) >= 5 else 0)
