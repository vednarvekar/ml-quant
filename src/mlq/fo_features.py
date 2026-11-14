"""Point-in-time futures facts and features built from the NSE F&O bhavcopy (audit step 2 + 4).

Outputs (all aligned to ctx.dates x ctx.syms):
  member     bool  stock futures existed on that date (point-in-time F&O list)
  lot_hist   float lot size on that date (new format: given; old format: inferred - open interest is
                   always a multiple of the lot size, so gcd of OI values, minimised over 15 days)
  oi_*       open-interest features (level change, price/OI build-up)
  basis_ann  annualised futures premium over spot (near-month settle vs NSE raw close)
  roll_flags bool  per day: roll positions now (2 trading days before each monthly expiry)
  open_x/close_x   ex-DIVIDEND price matrices (futures holders do not receive dividends)
"""
from math import gcd

import numpy as np
import polars as pl

from mlq.config import DATA

FO_DIR = DATA / "raw" / "nse_fo"
ROLL_DAYS = 2


def load_fo() -> pl.DataFrame:
    fs = sorted(FO_DIR.glob("*.parquet"))
    from mlq.backfill import apply_alias
    return apply_alias(pl.concat([pl.read_parquet(f) for f in fs]).filter(pl.col("kind") == "S"))


def _wide(df: pl.DataFrame, col: str, dates: list, syms: list) -> np.ndarray:
    w = df.pivot(on="sym", index="date", values=col, aggregate_function="first")
    out = np.full((len(dates), len(syms)), np.nan)
    ri = {d: i for i, d in enumerate(dates)}
    ci = {s: j for j, s in enumerate(syms)}
    rows = [ri.get(d) for d in w["date"].to_list()]
    for c in w.columns:
        if c == "date" or c not in ci:
            continue
        v = w[c].to_numpy().astype(float)
        for r, x in zip(rows, v):
            if r is not None:
                out[r, ci[c]] = x
    return out


def infer_lots(fo: pl.DataFrame) -> pl.DataFrame:
    """Per (date, sym) lot size: given (new format) or gcd of the OI numbers (old format)."""
    g = fo.group_by(["date", "sym"]).agg(pl.col("lot").max().alias("lot_given"),
                                         pl.col("oi").alias("oi"), pl.col("chg_oi").alias("chg"))
    lots = []
    for lg, oi, ch in g.select(["lot_given", "oi", "chg"]).iter_rows():
        if lg:
            lots.append(float(lg))
            continue
        v = 0
        for x in list(oi) + [abs(c) for c in ch if c]:
            if x:
                v = gcd(v, int(abs(x)))
        lots.append(float(v) if v else np.nan)
    g = g.select(["date", "sym"]).with_columns(pl.Series("lot_g", lots))
    return g.sort(["sym", "date"]).with_columns(
        pl.when(pl.col("lot_g") > 0).then(pl.col("lot_g")).otherwise(None).alias("lot_g")
    ).with_columns(pl.col("lot_g").rolling_min(15, min_samples=1).over("sym").alias("lot"))


def build(ctx, adj_dir=None, with_prices=True) -> dict:
    dates, syms = ctx.dates, ctx.syms
    fo = load_fo()
    T, N = len(dates), len(syms)
    f = {}
    present = fo.filter(pl.col("oi") > 0).select(["date", "sym"]).unique().with_columns(pl.lit(1.0).alias("m"))
    f["member"] = np.nan_to_num(_wide(present, "m", dates, syms), nan=0.0) > 0
    anym = f["member"].any(axis=1)
    if anym.any():                                      # dates after the last F&O file (not published yet):
        last = int(np.nonzero(anym)[0].max())           # assume yesterday's F&O list still applies
        f["member"][last + 1:] = f["member"][last]
    lots = infer_lots(fo)
    f["lot_hist"] = _wide(lots, "lot", dates, syms)
    # total OI and near-month contract
    tot = fo.group_by(["date", "sym"]).agg(pl.col("oi").sum().alias("oi_tot"),
                                           pl.col("contracts").sum().alias("fut_vol"))
    f["oi_tot"] = _wide(tot, "oi_tot", dates, syms)
    f["fut_vol"] = _wide(tot, "fut_vol", dates, syms)
    near = (fo.filter(pl.col("expiry") >= pl.col("date")).sort(["date", "sym", "expiry"])
            .group_by(["date", "sym"], maintain_order=True).first())
    near = near.with_columns((pl.col("expiry") - pl.col("date")).dt.total_days().alias("dte"))
    f["near_settle"] = _wide(near, "settle", dates, syms)
    f["near_dte"] = _wide(near, "dte", dates, syms)
    # features that need spot raw close (NSE unadjusted) for basis
    spot = np.full((T, N), np.nan)
    f["spot_raw"] = spot
    # roll flags: ROLL_DAYS trading days before each monthly expiry
    exp = sorted(set(fo["expiry"].drop_nulls().to_list()))
    di = {d: i for i, d in enumerate(dates)}
    flags = np.zeros(T, bool)
    for e in exp:
        j = di.get(e)
        if j is None:                                   # expiry on a holiday -> previous trading day
            prev = [i for i, d in enumerate(dates) if d <= e]
            j = prev[-1] if prev else None
        if j is not None and j - ROLL_DAYS >= 0 and e <= dates[-1]:
            flags[j - ROLL_DAYS] = True
    f["roll_flags"] = flags
    return f


def exdiv_prices(ctx, adj_dir=None, zero_flagged=False):
    """Open/close matrices with DIVIDENDS removed: use raw (unadjusted) returns on ordinary days,
    adjusted returns on split/bonus days (|raw - adjusted| large). Anchored at the last close."""
    adj_dir = adj_dir or DATA / "clean" / "adj_daily"
    T, N = len(ctx.dates), len(ctx.syms)
    ri = {d: i for i, d in enumerate(ctx.dates)}
    close_x = np.full((T, N), np.nan)
    open_x = np.full((T, N), np.nan)
    raw_close = np.full((T, N), np.nan)
    for j, s in enumerate(ctx.syms):
        f = adj_dir / f"{s}.parquet"
        if not f.exists():
            f = DATA / "clean" / "adj_daily_bf" / f"{s}.parquet"
        if not f.exists():
            continue
        d = pl.read_parquet(f, columns=["date", "close", "open", "ret", "ret_raw", "raw_close", "exclude_ret"]).sort("date")
        idx = np.array([ri[x] for x in d["date"].to_list() if x in ri])
        keep = np.array([x in ri for x in d["date"].to_list()])
        d = d.filter(pl.Series(keep))
        ret, rr = d["ret"].to_numpy(), d["ret_raw"].to_numpy()
        fr = np.where(np.isfinite(rr) & np.isfinite(ret) & (np.abs(rr - ret) < 0.03), rr, ret)
        fr = np.nan_to_num(fr, nan=0.0)
        if zero_flagged:                                  # diagnostic: remove unexplained jumps
            fr = np.where(d["exclude_ret"].fill_null(False).to_numpy() | (np.abs(fr) > 0.35), 0.0, fr)
        c_adj = d["close"].to_numpy()
        lvl = np.empty(len(fr))
        lvl[-1] = c_adj[-1]
        for k in range(len(fr) - 1, 0, -1):
            lvl[k - 1] = lvl[k] / (1 + fr[k])
        close_x[idx, j] = lvl
        open_x[idx, j] = d["open"].to_numpy() * lvl / c_adj
        raw_close[idx, j] = d["raw_close"].to_numpy()
    return open_x, close_x, raw_close


def ffill_lots(lot_hist: np.ndarray, limit: int = 60) -> np.ndarray:
    """Carry the last known lot size forward up to `limit` days (so a position can still be exited
    after a stock drops out of F&O); new entries are blocked separately by the membership mask."""
    out = lot_hist.copy()
    T, N = out.shape
    for j in range(N):
        last, age = np.nan, 10**9
        for t in range(T):
            if np.isfinite(out[t, j]):
                last, age = out[t, j], 0
            else:
                age += 1
                if np.isfinite(last) and age <= limit:
                    out[t, j] = last
    return out


def mto_matrix(ctx) -> np.ndarray:
    """Delivery % per (date, stock) from the NSE MTO archive; NaN where missing."""
    d = DATA / "raw" / "nse_mto"
    fs = sorted(d.glob("*.parquet"))
    if not fs:
        return np.full((len(ctx.dates), len(ctx.syms)), np.nan)
    from mlq.backfill import apply_alias
    df = apply_alias(pl.concat([pl.read_parquet(f, columns=["date", "sym", "deliv_pct"]) for f in fs]))
    df = df.unique(["date", "sym"], keep="first")
    return _wide(df, "deliv_pct", ctx.dates, ctx.syms)
