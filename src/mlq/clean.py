"""Clean raw Upstox candles -> data/clean/{1d,5m}/<sym>.parquet + flags report.

Fixes: daily duplicate dates (00:00 vs 09:15 stamps), non-positive OHLC rows,
marks special sessions (market-wide short days), and measures daily-vs-5m price
basis disagreement per symbol (adjustment inconsistency).

uv run python -m mlq.clean
"""
import polars as pl

from mlq.config import DATA
from mlq.quality import load_all, ohlc_violations

CLEAN = DATA / "clean"


def drop_stray_early(d: pl.DataFrame, gap_days: int = 365, min_rows: int = 60) -> pl.DataFrame:
    """Upstox sometimes returns a few stray rows years before a stock's real listing
    (e.g. SBICARD 2015 rows, listed 2020). Drop a leading segment of < min_rows rows that is
    followed by a gap of > gap_days."""
    d = d.sort(["sym", "date"]).with_columns(
        (pl.col("date") - pl.col("date").shift(1).over("sym")).dt.total_days().alias("_gap"))
    d = d.with_columns((pl.col("_gap") > gap_days).fill_null(False).cum_sum().over("sym").alias("_seg"))
    d = d.with_columns(pl.len().over(["sym", "_seg"]).alias("_n"))
    first_big = d.group_by("sym").agg(pl.col("_seg").max().alias("_maxseg"))
    d = d.join(first_big, on="sym")
    stray = (pl.col("_seg") == 0) & (pl.col("_maxseg") > 0) & (pl.col("_n") < min_rows)
    return d.filter(~stray).drop(["_gap", "_seg", "_n", "_maxseg"])


def clean_daily(d: pl.DataFrame) -> pl.DataFrame:
    d = d.with_columns(pl.col("ts").dt.date().alias("date"))
    # prefer the midnight-stamped row when a date has two
    d = d.sort(["sym", "date", "ts"]).unique(["sym", "date"], keep="first")
    bad = ohlc_violations(d)
    d = d.join(bad.select(["sym", "date"]), on=["sym", "date"], how="anti")
    d = drop_stray_early(d)
    return d.sort(["sym", "date"]).select(["sym", "date", "open", "high", "low", "close", "volume"])


def special_sessions(m: pl.DataFrame, min_syms: int = 30) -> pl.DataFrame:
    per = m.with_columns(pl.col("ts").dt.date().alias("date")) \
        .group_by(["sym", "date"]).agg(pl.len().alias("bars"))
    return (per.filter(pl.col("bars") < 70).group_by("date").agg(pl.len().alias("n_syms"))
            .filter(pl.col("n_syms") >= min_syms).sort("date"))


def clean_5m(m: pl.DataFrame, special: pl.DataFrame) -> pl.DataFrame:
    m = m.with_columns(pl.col("ts").dt.date().alias("date"))
    m = m.join(ohlc_violations(m).select(["sym", "ts"]), on=["sym", "ts"], how="anti")
    m = m.with_columns(pl.col("date").is_in(special["date"].implode()).alias("special_session"))
    return m.select(["sym", "ts", "date", "open", "high", "low", "close", "volume",
                     "special_session"]).sort(["sym", "ts"])


def basis_check(m: pl.DataFrame, d: pl.DataFrame) -> pl.DataFrame:
    """Per symbol: share of full 75-bar days where 5m last close differs from daily close >0.5%."""
    agg = m.filter(~pl.col("special_session")).sort("ts").group_by(["sym", "date"]).agg(
        pl.col("close").last().alias("c5"), pl.len().alias("bars")).filter(pl.col("bars") == 75)
    j = agg.join(d.select(["sym", "date", pl.col("close").alias("cd")]), on=["sym", "date"])
    j = j.with_columns((pl.col("c5") / pl.col("cd") - 1).abs().alias("gap"))
    return (j.group_by("sym").agg((pl.col("gap") > 0.005).mean().alias("share_mismatch"),
                                  pl.col("gap").median().alias("median_gap"),
                                  pl.col("gap").max().alias("max_gap"))
            .sort("share_mismatch", descending=True))


def clean_index_daily(d: pl.DataFrame, calendar: list) -> pl.DataFrame:
    """Upstox index dailies often have 2 rows/date: a close-only midnight row (O=H=L=C) and
    a 09:15 row with the real range (same close). Keep the widest-range row; keep only
    equity trading days (the bond index also prints weekends)."""
    d = d.with_columns(pl.col("ts").dt.date().alias("date"),
                       (pl.col("high") - pl.col("low")).alias("_rng"))
    d = d.sort(["date", "_rng"]).unique("date", keep="last").drop("_rng")
    d = d.filter(pl.col("date").is_in(calendar))
    return d.sort("date").select(["date", "open", "high", "low", "close", "volume"])


def clean_indices():
    import glob as _g
    from pathlib import Path as _P
    raw_root = DATA / "raw" / "upstox_idx"
    stocks = load_all("1d").with_columns(pl.col("ts").dt.date().alias("date"))
    cal = (stocks.group_by("date").agg(pl.len().alias("n")).filter(pl.col("n") >= 30)
           ["date"].to_list())
    for tf in ("1d", "5m"):
        out = CLEAN / "idx" / tf
        out.mkdir(parents=True, exist_ok=True)
        for f in _g.glob(str(raw_root / tf / "*.parquet")):
            d = pl.read_parquet(f)
            if tf == "1d":
                d = clean_index_daily(d, cal)
            else:
                d = d.unique("ts").sort("ts").with_columns(pl.col("ts").dt.date().alias("date"))
            d.write_parquet(out / _P(f).name)
    print("indices cleaned ->", CLEAN / "idx")


def main():
    clean_indices()
    raw_d, raw_m = load_all("1d"), load_all("5m")
    d = clean_daily(raw_d)
    sp = special_sessions(raw_m)
    m = clean_5m(raw_m, sp)
    for tf, df in (("1d", d), ("5m", m)):
        out = CLEAN / tf
        out.mkdir(parents=True, exist_ok=True)
        for (sym,), g in df.group_by("sym"):
            g.write_parquet(out / f"{sym}.parquet")
    bc = basis_check(m, d)
    bc.write_csv(CLEAN / "basis_check.csv")
    sp.write_csv(CLEAN / "special_sessions.csv")
    print(f"daily {raw_d.height}->{d.height} rows; 5m {raw_m.height}->{m.height} rows")
    print("special sessions (market-wide short days):")
    print(sp)
    print("symbols with >5% of days where 5m vs daily close differ >0.5%:",
          bc.filter(pl.col("share_mismatch") > 0.05).height, "of", bc.height)
    print(bc.head(15))


if __name__ == "__main__":
    main()
