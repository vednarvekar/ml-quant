"""NSE security-wise delivery position (MTO) - public static archive, one small file per trading day.

Delivery % = share of traded quantity that was actually delivered (not squared off intraday). High
delivery % on a rising day is read as accumulation. Saved to data/raw/nse_mto/{date}.parquet
(columns: date, sym, traded, deliv, deliv_pct) for series EQ only.

uv run python -m mlq.nse_mto
"""
import io
import sys
import time
from datetime import date

import httpx
import polars as pl

from mlq.config import DATA
from mlq.nse_bhav import trading_days

OUT = DATA / "raw" / "nse_mto"
H = {"User-Agent": "Mozilla/5.0"}
DELAY = 0.3


def url(d: date) -> str:
    return f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{d.strftime('%d%m%Y')}.DAT"


def parse(text: str, d: date) -> pl.DataFrame:
    rows = []
    for ln in text.splitlines():
        p = ln.strip().split(",")
        if len(p) >= 7 and p[0] == "20" and p[3].strip() == "EQ":
            try:
                rows.append((p[2].strip(), int(p[4]), int(p[5]), float(p[6])))
            except ValueError:
                continue
    df = pl.DataFrame(rows, schema=["sym", "traded", "deliv", "deliv_pct"], orient="row")
    return df.with_columns(pl.lit(d).alias("date")).select(["date", "sym", "traded", "deliv", "deliv_pct"])


def _fetch(c, d):
    for attempt in range(5):
        try:
            time.sleep(DELAY)
            r = c.get(url(d))
            break
        except httpx.TransportError:
            time.sleep(min(60, 5 * 2 ** attempt))
    else:
        return d, "netfail", None
    if r.status_code == 404:
        return d, "nofile", None
    r.raise_for_status()
    df = parse(r.text, d)
    return d, ("ok" if df.height else "empty"), df


def main(start: date = date(2012, 1, 1), workers: int = 4):
    from concurrent.futures import ThreadPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    nf = OUT / "_no_file.txt"
    no_file = set(nf.read_text().split()) if nf.exists() else set()
    days = [d for d in trading_days() if d >= start and d.isoformat() not in no_file
            and not (OUT / f"{d}.parquet").exists()]
    print(f"MTO: {len(days)} days to fetch", flush=True)
    fails = 0
    with httpx.Client(headers=H, timeout=60, follow_redirects=True,
                      limits=httpx.Limits(max_connections=workers)) as c, ThreadPoolExecutor(workers) as ex:
        for i, (d, st, df) in enumerate(ex.map(lambda x: _fetch(c, x), days), 1):
            if st == "ok":
                df.write_parquet(OUT / f"{d}.parquet")
                fails = 0
            elif st in ("nofile", "empty"):
                no_file.add(d.isoformat())
                nf.write_text("\n".join(sorted(no_file)))
            elif st == "netfail":
                fails += 1
                if fails >= 8:
                    sys.exit("network down; rerun to resume")
            if i % 200 == 0:
                print(f"  {i}/{len(days)} {d}", flush=True)
    print("MTO DONE", flush=True)


if __name__ == "__main__":
    main()
