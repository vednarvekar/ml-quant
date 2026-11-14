"""Refresh data for paper trading: new daily candles (read-only Upstox analytics token), NSE
bhavcopy, then the offline rebuild chain. Rate-limit safe (1 request/second, resumable).

uv run python -m mlq.paper.update
"""
import os
import sys
import time
from datetime import date
from pathlib import Path

from mlq.config import DATA, INSTRUMENTS_FILE, RAW
from mlq.upstox.auth import load_token
from mlq.upstox.history import Throttle, download
from mlq.upstox.instruments import equities, refresh


def refresh_data(refresh_instruments_days: int = 7) -> dict:
    # instrument master (lot sizes, keys): weekly is plenty
    if (not INSTRUMENTS_FILE.exists()
            or time.time() - os.path.getmtime(INSTRUMENTS_FILE) > refresh_instruments_days * 86400):
        refresh()
    import polars as pl
    eq = equities()
    keys = dict(eq.select(["trading_symbol", "instrument_key"]).iter_rows())
    idx = pl.read_parquet(INSTRUMENTS_FILE).filter(pl.col("segment") == "NSE_INDEX")
    idx_keys = dict(idx.select(["name", "instrument_key"]).iter_rows())
    syms = sorted(Path(f).stem for f in (RAW / "1d").glob("*.parquet"))
    token, th, end = load_token(), Throttle(1.0), date.today()
    failed, skipped, streak = [], [], 0
    for sym in syms:
        if sym not in keys:
            skipped.append(sym)
            continue
        try:
            download(sym, keys[sym], "days", 1, date(2005, 1, 1), end, token, th)
            streak = 0
        except Exception as e:                                   # noqa: BLE001
            failed.append((sym, str(e)[:80]))
            streak += 1
            if "401" in str(e):
                sys.exit("Token rejected (401): update UPSTOX_ANALYTICS_TOKEN in .env")
            if streak >= 3:
                sys.exit("3 failures in a row (network?); rerun to resume")
    root = DATA / "raw" / "upstox_idx"
    for name in ("Nifty 50", "India VIX"):
        try:
            download(name.replace(" ", "_"), idx_keys[name], "days", 1, date(2005, 1, 1), end, token, th, root=root)
        except Exception as e:                                   # noqa: BLE001
            failed.append((name, str(e)[:80]))
    # 1-minute candles of the live stock-futures contracts (real fill prices; lost once a contract
    # expires, so archive every evening). Never blocks the paper run.
    try:
        from mlq.upstox import fut_minutes
        fut = fut_minutes.refresh(0.25)
        fut = {"contracts": fut["contracts"], "failed": len(fut["failed"])}
    except Exception as e:                                       # noqa: BLE001
        fut = {"error": str(e)[:80]}
    return {"symbols": len(syms), "skipped_no_key": skipped, "failed": failed, "futures_minutes": fut}


def rebuild_offline() -> None:
    from mlq import adjust, clean, nse_bhav, nse_fo, nse_mto, review, universe
    nse_bhav.main()
    nse_fo.main()          # futures bhavcopy (membership, OI, basis, expiry, lots)
    nse_mto.main()         # delivery %
    clean.main()
    adjust.main()
    review.main()
    universe.main()


if __name__ == "__main__":
    r = refresh_data()
    print(r)
    rebuild_offline()
