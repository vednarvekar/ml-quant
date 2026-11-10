"""Download Upstox NSE instrument master and keep equities."""
import gzip
import io
import json

import httpx
import polars as pl

from mlq.config import INSTRUMENTS_FILE

URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"


def refresh() -> pl.DataFrame:
    r = httpx.get(URL, timeout=60, follow_redirects=True)
    r.raise_for_status()
    rows = json.loads(gzip.GzipFile(fileobj=io.BytesIO(r.content)).read())
    df = pl.DataFrame(rows, infer_schema_length=None)
    INSTRUMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(INSTRUMENTS_FILE)
    return df


def equities(df: pl.DataFrame | None = None) -> pl.DataFrame:
    df = df if df is not None else pl.read_parquet(INSTRUMENTS_FILE)
    return df.filter((pl.col("segment") == "NSE_EQ") & (pl.col("instrument_type") == "EQ"))


def key_for(symbol: str, df: pl.DataFrame | None = None) -> str:
    eq = equities(df).filter(pl.col("trading_symbol") == symbol)
    if eq.is_empty():
        raise KeyError(symbol)
    return eq["instrument_key"][0]


if __name__ == "__main__":
    d = refresh()
    print(d.height, "instruments;", equities(d).height, "NSE equities")
