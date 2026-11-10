"""Historical candle downloader (Upstox V3) with chunking, throttling, resume.

Limits (verified from Upstox docs): 1-15 min -> 1 month/request, >15 min and hours
-> 1 quarter, days -> 1 decade. Rate: 2000 req / 30 min -> we cap ~1 req / 1.0 s.
"""
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import httpx
import polars as pl

from mlq.config import RAW

BASE = "https://api.upstox.com/v3/historical-candle"
MIN_INTERVAL_S = 1.0  # ~1800 req / 30 min, under the 2000 cap


def window_days(unit: str, interval: int) -> int:
    """Safe window size in days, slightly under the documented limits."""
    if unit == "minutes":
        return 28 if interval <= 15 else 85
    if unit == "hours":
        return 85
    if unit == "days":
        return 3500
    return 36500  # weeks / months: no limit


def chunks(start: date, end: date, unit: str, interval: int):
    """Yield (from_date, to_date) windows covering [start, end], oldest first."""
    step = timedelta(days=window_days(unit, interval))
    cur = start
    while cur <= end:
        nxt = min(cur + step - timedelta(days=1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


SCHEMA = {"ts": pl.Datetime("us", "Asia/Kolkata"), "open": pl.Float64, "high": pl.Float64,
          "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64, "oi": pl.Int64}


def parse_candles(candles: list) -> pl.DataFrame:
    """candles: [[iso_ts, o, h, l, c, v, oi], ...] (API returns newest first)."""
    if not candles:
        return pl.DataFrame(schema=SCHEMA)
    rows = [(c[0], float(c[1]), float(c[2]), float(c[3]), float(c[4]),
             int(c[5]), int(c[6]) if len(c) > 6 and c[6] is not None else 0) for c in candles]
    df = pl.DataFrame(rows, schema=["ts", "open", "high", "low", "close", "volume", "oi"],
                      orient="row")
    df = df.with_columns(
        pl.col("ts").str.to_datetime("%Y-%m-%dT%H:%M:%S%z").dt.convert_time_zone("Asia/Kolkata")
    )
    return df.unique("ts").sort("ts")


class Throttle:
    def __init__(self, min_interval: float = MIN_INTERVAL_S):
        self.min_interval, self._last = min_interval, 0.0

    def wait(self):
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.monotonic()


def fetch_window(client: httpx.Client, token: str, key: str, unit: str, interval: int,
                 frm: date, to: date, throttle: Throttle) -> pl.DataFrame:
    url = f"{BASE}/{quote(key, safe='')}/{unit}/{interval}/{to.isoformat()}/{frm.isoformat()}"
    for attempt in range(6):
        throttle.wait()
        try:
            r = client.get(url, headers={"Authorization": f"Bearer {token}",
                                         "accept": "application/json"})
        except httpx.TransportError:  # disconnect, DNS, timeout: back off and retry
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return parse_candles(r.json().get("data", {}).get("candles", []))
    raise RuntimeError(f"gave up after retries (rate limit or network) for {key}")


def out_path(symbol: str, unit: str, interval: int, root: Path = RAW) -> Path:
    return root / f"{interval}{unit[0]}" / f"{symbol}.parquet"


def missing_ranges(existing_min: date | None, existing_max: date | None,
                   start: date, end: date) -> list[tuple[date, date]]:
    """Ranges still to fetch given what's stored (head gap + tail extension)."""
    if existing_min is None:
        return [(start, end)]
    out = []
    if start < existing_min:
        out.append((start, existing_min))
    if existing_max < end:
        out.append((existing_max, end))
    return out


def download(symbol: str, key: str, unit: str, interval: int, start: date, end: date,
             token: str, throttle: Throttle | None = None, root: Path = RAW) -> Path:
    """Download [start, end], merging with any existing file (resume-friendly)."""
    path = out_path(symbol, unit, interval, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=SCHEMA)
    lo = existing["ts"].min().date() if existing.height else None
    hi = existing["ts"].max().date() if existing.height else None
    parts, th = [existing], throttle or Throttle()
    with httpx.Client(timeout=30) as client:
        for a, b in missing_ranges(lo, hi, start, end):
            for frm, to in chunks(a, b, unit, interval):
                parts.append(fetch_window(client, token, key, unit, interval, frm, to, th))
    parts = [p for p in parts if p.height]
    if parts:
        df = pl.concat(parts, how="vertical_relaxed").unique("ts").sort("ts")
        df.write_parquet(path)
    return path
