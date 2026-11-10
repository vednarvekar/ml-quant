"""Context data that isn't on Upstox.

1) FRED (US Federal Reserve public data, official CSV endpoint): USDINR, Brent crude,
   S&P 500, Nasdaq, US VIX, US 10y yield. US markets close AFTER India, so a US value dated
   D may only be used for Indian trading on D+1 (lag it in features; never same-day).
2) NSE participant-wise open interest (public archive CSV, one per trading day): how many
   index/stock futures & options contracts Clients, DII, FII and Pro hold long/short.
   This is our "big money positioning" signal (NSE's FII/DII cash-flow API blocks scripts).

uv run python -m mlq.context
"""
import io
import sys
import time
from datetime import date

import httpx
import polars as pl

from mlq.config import DATA
from mlq.nse_bhav import trading_days

FRED_DIR = DATA / "raw" / "fred"
POI_DIR = DATA / "raw" / "nse_participant_oi"
H = {"User-Agent": "Mozilla/5.0"}
FRED = {"DEXINUS": "usdinr", "DCOILBRENTEU": "brent", "SP500": "sp500",
        "NASDAQCOM": "nasdaq", "VIXCLS": "us_vix", "DGS10": "us10y"}


def fetch_fred():
    FRED_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=H, timeout=60) as c:
        for sid, name in FRED.items():
            r = c.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid})
            r.raise_for_status()
            df = pl.read_csv(io.BytesIO(r.content), infer_schema_length=0)
            df = df.select(pl.col(df.columns[0]).str.to_date().alias("date"),
                           pl.col(df.columns[1]).cast(pl.Float64, strict=False).alias(name))
            df = df.filter(pl.col(name).is_not_null())
            df.write_parquet(FRED_DIR / f"{name}.parquet")
            print(f"FRED {sid}->{name}: {df.height} rows {df['date'].min()}..{df['date'].max()}",
                  flush=True)
            time.sleep(1)


def parse_poi(text: str, d: date) -> pl.DataFrame:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    hdr_i = next(i for i, ln in enumerate(lines)
                 if ln.strip('"').lower().replace("_", " ").startswith("client type"))
    df = pl.read_csv(io.StringIO("\n".join(lines[hdr_i:])), infer_schema_length=0)

    def norm(c: str) -> str:  # old files: CLIENT_TYPE / "Total Long Contract"
        c = c.strip().strip('"').lower().replace(" ", "_")
        return c + "s" if c.endswith("_contract") else c

    df = df.rename({c: norm(c) for c in df.columns})
    df = df.select([c for c in df.columns if c and not c.startswith("_")])
    num = [c for c in df.columns if c != "client_type"]
    return df.with_columns([pl.col(c).str.strip_chars().str.replace_all(",", "")
                            .cast(pl.Int64, strict=False) for c in num]) \
             .with_columns(pl.col("client_type").str.strip_chars().str.to_uppercase(),
                           pl.lit(d).alias("date"))


def fetch_participant_oi(start: date = date(2012, 1, 1)):
    POI_DIR.mkdir(parents=True, exist_ok=True)
    nofile_f = POI_DIR / "_no_file.txt"
    nofile = set(nofile_f.read_text().split()) if nofile_f.exists() else set()
    days = [d for d in trading_days() if d >= start and d.isoformat() not in nofile
            and not (POI_DIR / f"{d}.parquet").exists()]
    print(f"participant OI: {len(days)} days to fetch", flush=True)
    streak = 0
    with httpx.Client(headers=H, timeout=30, follow_redirects=True) as c:
        for i, d in enumerate(days, 1):
            url = ("https://nsearchives.nseindia.com/content/nsccl/"
                   f"fao_participant_oi_{d.strftime('%d%m%Y')}.csv")
            try:
                time.sleep(0.4)
                r = c.get(url)
                streak = 0
            except httpx.TransportError:
                streak += 1
                if streak >= 3:
                    sys.exit("network down; rerun to resume")
                time.sleep(10)
                continue
            if r.status_code == 404:
                nofile.add(d.isoformat())
                nofile_f.write_text("\n".join(sorted(nofile)))
                continue
            r.raise_for_status()
            try:
                parse_poi(r.text, d).write_parquet(POI_DIR / f"{d}.parquet")
            except (StopIteration, pl.exceptions.PolarsError) as e:  # odd format: keep raw
                bad = POI_DIR / "_unparsed"
                bad.mkdir(exist_ok=True)
                (bad / f"{d}.csv").write_bytes(r.content)
                print(f"  unparsed {d}: {type(e).__name__}", flush=True)
            if i % 250 == 0:
                print(f"  {i}/{len(days)} {d}", flush=True)
    print("participant OI DONE", flush=True)


if __name__ == "__main__":
    fetch_fred()
    fetch_participant_oi()
