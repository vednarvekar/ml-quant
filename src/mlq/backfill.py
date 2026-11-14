"""Backfill: daily prices for stocks we have NO Upstox history for (mostly delisted / merged /
shrunk companies), built from the NSE cash bhavcopy, so the 2013-2021 holdout is not
survivor-biased. Research-only: used when MLQ_BACKFILL=1; the live paper trader never sees it.

Corporate actions (no free corporate-action feed): NSE raw prices are unadjusted, so a split or
bonus shows as an overnight drop by a standard ratio m (2, 3, 5, 10, 1.5 ...). We adjust when
  * the gap close[t-1]/open[t] is within 6% of a standard m, AND
  * futures open interest (in shares) jumps by ~m the same day (NSE adjusts F&O on these events),
    or, with no F&O data that day, m >= 1.45 (too big to be an ordinary move).
Other moves larger than 35% stay as-is and are flagged (ret_class 'bf_unexplained').
Dividends are NOT adjusted (same basis as the futures P&L, which uses ex-dividend prices).

Renames: old NSE symbols are mapped to the entity's latest symbol (ISIN-linked), for prices,
F&O membership/lots and delivery %.

uv run python -m mlq.backfill
"""
import os

import numpy as np
import polars as pl

from mlq.adjust import ALIASES, load_bhav
from mlq.config import DATA
from mlq.universe import entity_map, fund_symbols

OUT = DATA / "clean" / "adj_daily_bf"
RATIOS = [1.25, 4 / 3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0]
FROM = "2008-01-01"


def enabled() -> bool:
    return os.environ.get("MLQ_BACKFILL") == "1"


def alias_map() -> dict:
    """old NSE symbol -> latest symbol of the same entity (only where they differ)."""
    f = DATA / "clean" / "symbol_alias.parquet"
    if not f.exists():
        return {}
    d = pl.read_parquet(f)
    return dict(zip(d["nse_symbol"].to_list(), d["name"].to_list()))


def apply_alias(df: pl.DataFrame, col: str = "sym") -> pl.DataFrame:
    """Rename old symbols to the latest name (only when backfill is enabled)."""
    if not enabled():
        return df
    am = alias_map()
    return df.with_columns(pl.col(col).replace(am)) if am else df


def fo_oi() -> pl.DataFrame:
    fs = sorted((DATA / "raw" / "nse_fo").glob("*.parquet"))
    fo = pl.concat([pl.read_parquet(f, columns=["date", "sym", "kind", "oi"]) for f in fs])
    return fo.filter(pl.col("kind") == "S").group_by(["date", "sym"]).agg(pl.col("oi").sum())


def adjust_one(g: pl.DataFrame, oi: dict) -> tuple[pl.DataFrame, list]:
    """g: raw daily rows of one entity (date-sorted). oi: date -> futures OI. Returns adjusted
    frame and the list of applied events."""
    c = g["close"].to_numpy().astype(float)
    o = g["open"].to_numpy().astype(float)
    dates = g["date"].to_list()
    fac = np.ones(len(c))                       # multiply prices on/before t-1 by 1/m
    events = []
    for t in range(1, len(c)):
        if not (c[t - 1] > 0 and o[t] > 0):
            continue
        m_obs = c[t - 1] / o[t]                 # overnight gap (open-based)
        m_cls = c[t - 1] / c[t] if c[t] > 0 else np.nan
        if m_obs < 1.2 or not (np.isfinite(m_cls) and abs(m_cls / m_obs - 1) < 0.2):
            continue                            # a bad open print, not a corporate action
        m = min(RATIOS, key=lambda r: abs(m_obs / r - 1))
        a, b = oi.get(dates[t - 1]), oi.get(dates[t])
        if a and b and abs((b / a) / m_obs - 1) < 0.15:
            # futures OI (shares) scaled with the price -> NSE adjusted the contract: a split,
            # bonus or demerger. Use the standard ratio if close, else the observed gap.
            if abs(m_obs / m - 1) > 0.08:
                m = m_obs
        elif a and b and b / a > 1.3 and abs(m_obs / m - 1) < 0.06:
            pass                                # standard ratio + OI clearly scaled up
        elif not (a and b) and m >= 1.45 and abs(m_obs / m - 1) < 0.06:
            pass                                # no F&O that day: only clear standard ratios
        else:
            continue
        if True:
            fac[:t] /= m
            events.append((dates[t], m))
    vol_fac = 1 / fac
    out = g.with_columns(
        pl.col("close").alias("raw_close"), pl.col("prev_close").alias("raw_prev_close"),
        (pl.col("open") * fac).alias("open"), (pl.col("high") * fac).alias("high"),
        (pl.col("low") * fac).alias("low"), (pl.col("close") * fac).alias("close"),
        (pl.col("volume") * vol_fac).round(0).cast(pl.Int64).alias("volume"))
    out = out.with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret"),
        (pl.col("raw_close") / pl.col("raw_prev_close") - 1).alias("ret_raw"))
    ev = {d for d, _ in events}
    out = out.with_columns(
        pl.col("date").is_in(list(ev)).alias("adj_step"),
        (pl.col("ret").abs() > 0.25).fill_null(False).alias("review"),
        pl.lit(False).alias("manual_event"))
    out = out.with_columns(
        pl.when(pl.col("ret").abs() > 0.35).then(pl.lit("bf_unexplained")).otherwise(pl.lit("normal"))
        .alias("ret_class"))
    out = out.with_columns((pl.col("ret_class") == "bf_unexplained").alias("exclude_ret"))
    cols = ["date", "close", "open", "high", "low", "volume", "raw_close", "raw_prev_close", "ret",
            "ret_raw", "adj_step", "review", "manual_event", "ret_class", "exclude_ret"]
    return out.select(cols), events


def main():
    b = load_bhav()
    b = b.filter(~pl.col("nse_symbol").is_in(list(fund_symbols(b))))
    em = entity_map(b)
    old2new = {v: k for k, v in ALIASES.items()}           # renames after NSE dropped ISIN
    em = em.with_columns(pl.col("name").replace(old2new))
    em.filter(pl.col("nse_symbol") != pl.col("name")).select(["nse_symbol", "name"]) \
        .write_parquet(DATA / "clean" / "symbol_alias.parquet")
    have = {f.stem for f in (DATA / "clean" / "adj_daily").glob("*.parquet")}
    uni = pl.read_parquet(DATA / "clean" / "universe.parquet")
    want_uni = set(uni.filter((pl.col("rank") <= 200) & (pl.col("month") < pl.date(2022, 1, 1)))["name"].to_list())
    oi = fo_oi()
    sym2name = dict(zip(em["nse_symbol"].to_list(), em["name"].to_list()))
    oi = oi.with_columns(pl.col("sym").replace(sym2name).alias("name")) \
        .group_by(["date", "name"]).agg(pl.col("oi").sum())
    want_fo = set(oi.filter(pl.col("date") < pl.date(2022, 1, 1))["name"].unique().to_list())
    want = sorted((want_uni | want_fo) - have - set(old2new))
    print(f"entities to backfill: {len(want)} (universe {len(want_uni - have)}, F&O {len(want_fo - have)})",
          flush=True)
    bb = (b.join(em.select(["nse_symbol", "name"]), on="nse_symbol")
          .filter(pl.col("name").is_in(want) & (pl.col("date") >= pl.lit(FROM).str.to_date()))
          .filter((pl.col("close") > 0) & (pl.col("prev_close") > 0))
          .sort(["name", "date", "volume"]).unique(["name", "date"], keep="last").sort(["name", "date"]))
    oi_by = {k[0]: dict(zip(g["date"].to_list(), g["oi"].to_list()))
             for k, g in oi.filter(pl.col("name").is_in(want)).group_by("name")}
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.parquet"):
        f.unlink()
    n_ev, n_unx, rows = 0, 0, []
    for (name,), g in bb.group_by("name"):
        if g.height < 120:
            continue
        out, ev = adjust_one(g.sort("date"), oi_by.get(name, {}))
        out.write_parquet(OUT / f"{name}.parquet")
        n_ev += len(ev)
        n_unx += int(out["exclude_ret"].sum())
        rows += [{"sym": name, "date": d, "ratio": m} for d, m in ev]
    pl.DataFrame(rows, schema={"sym": pl.Utf8, "date": pl.Date, "ratio": pl.Float64}) \
        .write_csv(DATA / "clean" / "backfill_events.csv")
    print(f"wrote {len(list(OUT.glob('*.parquet')))} symbols | split/bonus adjustments {n_ev} | "
          f"unexplained >35% moves {n_unx} -> {OUT}")


if __name__ == "__main__":
    main()
