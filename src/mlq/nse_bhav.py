"""Download NSE equity bhavcopy (unadjusted EOD, with PREVCLOSE).

Public static archive files (nsearchives). Polite: ~0.4s between requests, resumable.
All EQ rows kept (old files lack ISIN, symbols get renamed; we map to our universe later).
One parquet per trading day.

uv run python -m mlq.nse_bhav
"""
import io
import sys
import time
import zipfile
from datetime import date

import httpx
import polars as pl

from mlq.config import DATA
from mlq.quality import load_all

OUT = DATA / "raw" / "nse_bhav"
H = {"User-Agent": "Mozilla/5.0"}
DELAY = 0.4


def url(d: date) -> str:
    return (f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{d.year}/"
            f"{d.strftime('%b').upper()}/cm{d.strftime('%d%b%Y').upper()}bhav.csv.zip")


def parse(content: bytes, isins: set[str]) -> pl.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(content))
    df = pl.read_csv(io.BytesIO(z.read(z.namelist()[0])), infer_schema_length=0)
    df = df.rename({c: c.strip() for c in df.columns}).filter(pl.col("SERIES") == "EQ")
    if "ISIN" not in df.columns:  # pre-~2010 files have no ISIN; map by symbol later
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("ISIN"))
    return df.select(
        pl.col("SYMBOL").str.strip_chars().alias("nse_symbol"), "ISIN",
        pl.col("OPEN").cast(pl.Float64).alias("open"), pl.col("HIGH").cast(pl.Float64).alias("high"),
        pl.col("LOW").cast(pl.Float64).alias("low"), pl.col("CLOSE").cast(pl.Float64).alias("close"),
        pl.col("PREVCLOSE").cast(pl.Float64).alias("prev_close"),
        pl.col("TOTTRDQTY").cast(pl.Int64).alias("volume"))


def url_new(d: date) -> str:
    return f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d.strftime('%d%m%Y')}.csv"


def parse_new(content: bytes) -> pl.DataFrame:
    df = pl.read_csv(io.BytesIO(content), infer_schema_length=0)
    df = df.rename({c: c.strip() for c in df.columns})
    df = df.with_columns(pl.col("SERIES").str.strip_chars()).filter(pl.col("SERIES") == "EQ")
    num = lambda c: pl.col(c).str.strip_chars().cast(pl.Float64, strict=False)
    return df.select(
        pl.col("SYMBOL").str.strip_chars().alias("nse_symbol"),
        pl.lit(None, dtype=pl.Utf8).alias("ISIN"),
        num("OPEN_PRICE").alias("open"), num("HIGH_PRICE").alias("high"),
        num("LOW_PRICE").alias("low"), num("CLOSE_PRICE").alias("close"),
        num("PREV_CLOSE").alias("prev_close"),
        pl.col("TTL_TRD_QNTY").str.strip_chars().cast(pl.Int64, strict=False).alias("volume"))


def trading_days() -> list[date]:
    d = load_all("1d").with_columns(pl.col("ts").dt.date().alias("date"))
    cal = d.group_by("date").agg(pl.len().alias("n")).filter(pl.col("n") >= 30)
    return sorted(cal["date"].to_list())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wl = pl.read_csv(DATA / "nifty100_list.csv")
    isins = set(wl["ISIN Code"].to_list())
    holidays_file = OUT / "_no_file.txt"
    no_file = set(holidays_file.read_text().split()) if holidays_file.exists() else set()
    days = [d for d in trading_days() if not (OUT / f"{d}.parquet").exists()
            and d.isoformat() not in no_file]
    print(f"{len(days)} days to fetch", flush=True)
    streak = 0
    with httpx.Client(headers=H, timeout=30, follow_redirects=True) as c:
        for i, d in enumerate(days, 1):
            for attempt in range(5):
                try:
                    time.sleep(DELAY)
                    r = c.get(url(d))
                    break
                except httpx.TransportError:
                    time.sleep(min(60, 5 * 2 ** attempt))
            else:
                streak += 1
                print("network failure", d, flush=True)
                if streak >= 3:
                    sys.exit("network down; rerun to resume")
                continue
            streak = 0
            new_fmt = False
            if r.status_code == 404:  # mid-2024+: files moved to sec_bhavdata_full
                time.sleep(DELAY)
                r = c.get(url_new(d))
                new_fmt = True
            if r.status_code == 404:
                no_file.add(d.isoformat())  # e.g. weekend special session: no bhavcopy
                holidays_file.write_text("\n".join(sorted(no_file)))
                continue
            r.raise_for_status()
            out = parse_new(r.content) if new_fmt else parse(r.content, isins)
            out.write_parquet(OUT / f"{d}.parquet")
            if i % 200 == 0:
                print(f"{i}/{len(days)} {d}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
