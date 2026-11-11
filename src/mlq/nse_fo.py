"""NSE F&O bhavcopy (free public archive): daily stock + index FUTURES rows only.

Gives us what Upstox spot candles cannot: which stocks had futures on each date (point-in-time
F&O membership), real futures settlement prices, open interest and OI change, expiry dates
(for roll timing) and, from mid-2024, the lot size.

Old format (to Jun 2024): fo{DDMONYYYY}bhav.csv.zip ; new UDiFF format (Jul 2024+):
BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip. Saved to data/raw/nse_fo/{date}.parquet.

uv run python -m mlq.nse_fo
"""
import io
import sys
import time
import zipfile
from datetime import date

import httpx
import polars as pl

from mlq.config import DATA
from mlq.nse_bhav import trading_days

OUT = DATA / "raw" / "nse_fo"
H = {"User-Agent": "Mozilla/5.0"}
DELAY = 0.3
NEW_FROM = date(2024, 7, 8)
SCHEMA = {"date": pl.Date, "sym": pl.String, "kind": pl.String, "expiry": pl.Date, "open": pl.Float64,
          "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "settle": pl.Float64,
          "contracts": pl.Int64, "oi": pl.Int64, "chg_oi": pl.Int64, "lot": pl.Int64}


def url_old(d: date) -> str:
    return ("https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
            f"{d.year}/{d.strftime('%b').upper()}/fo{d.strftime('%d%b%Y').upper()}bhav.csv.zip")


def url_new(d: date) -> str:
    return f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"


def _read(content: bytes) -> pl.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(content))
    return pl.read_csv(io.BytesIO(z.read(z.namelist()[0])), infer_schema_length=0, truncate_ragged_lines=True)


def _num(c, dt=pl.Float64):
    return pl.col(c).str.strip_chars().str.replace_all(",", "").cast(dt, strict=False)


def parse_old(content: bytes, d: date) -> pl.DataFrame:
    df = _read(content)
    df = df.rename({c: c.strip() for c in df.columns})
    df = df.filter(pl.col("INSTRUMENT").str.strip_chars().is_in(["FUTSTK", "FUTIDX"]))
    return df.select(
        pl.lit(d).alias("date"), pl.col("SYMBOL").str.strip_chars().alias("sym"),
        pl.when(pl.col("INSTRUMENT").str.strip_chars() == "FUTSTK").then(pl.lit("S")).otherwise(pl.lit("I")).alias("kind"),
        pl.col("EXPIRY_DT").str.strip_chars().str.to_date("%d-%b-%Y", strict=False).alias("expiry"),
        _num("OPEN").alias("open"), _num("HIGH").alias("high"), _num("LOW").alias("low"),
        _num("CLOSE").alias("close"), _num("SETTLE_PR").alias("settle"),
        _num("CONTRACTS", pl.Int64).alias("contracts"), _num("OPEN_INT", pl.Int64).alias("oi"),
        _num("CHG_IN_OI", pl.Int64).alias("chg_oi"), pl.lit(None, dtype=pl.Int64).alias("lot"))


def parse_new(content: bytes, d: date) -> pl.DataFrame:
    df = _read(content)
    df = df.rename({c: c.strip() for c in df.columns})
    df = df.filter(pl.col("FinInstrmTp").str.strip_chars().is_in(["STF", "IDF"]))
    lot = "NewBrdLotQty" if "NewBrdLotQty" in df.columns else None
    return df.select(
        pl.lit(d).alias("date"), pl.col("TckrSymb").str.strip_chars().alias("sym"),
        pl.when(pl.col("FinInstrmTp").str.strip_chars() == "STF").then(pl.lit("S")).otherwise(pl.lit("I")).alias("kind"),
        pl.col("XpryDt").str.strip_chars().str.to_date("%Y-%m-%d", strict=False).alias("expiry"),
        _num("OpnPric").alias("open"), _num("HghPric").alias("high"), _num("LwPric").alias("low"),
        _num("ClsPric").alias("close"), _num("SttlmPric").alias("settle"),
        _num("TtlTradgVol", pl.Int64).alias("contracts"), _num("OpnIntrst", pl.Int64).alias("oi"),
        _num("ChngInOpnIntrst", pl.Int64).alias("chg_oi"),
        (_num(lot, pl.Int64) if lot else pl.lit(None, dtype=pl.Int64)).alias("lot"))


def _fetch(c: httpx.Client, d: date):
    """Returns (date, status, DataFrame|None). status: ok | nofile | netfail | parsefail."""
    for attempt in range(5):
        try:
            time.sleep(DELAY)
            r = c.get(url_new(d) if d >= NEW_FROM else url_old(d))
            break
        except httpx.TransportError:
            time.sleep(min(60, 5 * 2 ** attempt))
    else:
        return d, "netfail", None
    if r.status_code == 404:
        return d, "nofile", None
    r.raise_for_status()
    try:
        return d, "ok", (parse_new if d >= NEW_FROM else parse_old)(r.content, d)
    except Exception as e:                                         # noqa: BLE001
        print(f"  parse fail {d}: {type(e).__name__} {str(e)[:80]}", flush=True)
        return d, "parsefail", None


def main(start: date = date(2012, 1, 1), workers: int = 4):
    from concurrent.futures import ThreadPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    nf = OUT / "_no_file.txt"
    no_file = set(nf.read_text().split()) if nf.exists() else set()
    days = [d for d in trading_days() if d >= start and d.isoformat() not in no_file
            and not (OUT / f"{d}.parquet").exists()]
    print(f"F&O bhavcopy: {len(days)} days to fetch ({workers} workers)", flush=True)
    fails = 0
    with httpx.Client(headers=H, timeout=60, follow_redirects=True,
                      limits=httpx.Limits(max_connections=workers)) as c, ThreadPoolExecutor(workers) as ex:
        for i, (d, status, df) in enumerate(ex.map(lambda x: _fetch(c, x), days), 1):
            if status == "ok":
                df.cast(SCHEMA).write_parquet(OUT / f"{d}.parquet")
                fails = 0
            elif status == "nofile":
                no_file.add(d.isoformat())
                nf.write_text("\n".join(sorted(no_file)))
            elif status == "netfail":
                fails += 1
                if fails >= 8:
                    sys.exit("network down; rerun to resume")
            if i % 200 == 0:
                print(f"  {i}/{len(days)} {d}", flush=True)
    print("F&O DONE", flush=True)


if __name__ == "__main__":
    main()
