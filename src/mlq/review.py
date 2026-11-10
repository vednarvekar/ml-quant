"""Classify |ret| > 25% daily moves and flag the ones that are NOT tradable return signal.

Classes (first match wins):
  demerger_unadjusted   manual list in adjust.py (Upstox left the demerger drop in)
  tick_quantization     adjusted price < 20 Rs: Upstox's heavily adjusted early prices sit on a
                        0.05 tick grid, so +-33% / +-50% / +100% "moves" are rounding artifacts
  market_wide           median universe return beyond +-7% or >= 8 stocks moved >10% that day
  sector_group          3-7 stocks moved >10% the same day (PSU banks 2017-10-25, Adani 2023)
  suspected_unadjusted  |ret| > 60% on a non-market day: almost surely a corporate action
                        Upstox did not adjust (CGPOWER 2015/2016, ADANIENT 2015)
  news_vol_spike        volume >= 2.5x its 20-day mean: plausible real event
  unclear

`exclude_ret` = demerger_unadjusted | tick_quantization | suspected_unadjusted. Training code
should drop those stock-days from return labels. Adds ret_class / exclude_ret to adj_daily.

uv run python -m mlq.review
"""
import glob
from pathlib import Path

import polars as pl

from mlq.config import DATA

EXCLUDE = ["demerger_unadjusted", "tick_quantization", "suspected_unadjusted"]


def classify(d: pl.DataFrame) -> pl.DataFrame:
    """d: sym,date,close,volume,ret,review,manual_event (sorted by sym,date)."""
    d = d.sort(["sym", "date"]).with_columns(
        (pl.col("volume") / pl.col("volume").rolling_mean(20).shift(1).over("sym")).alias("vol_x"),
        pl.col("close").shift(1).over("sym").alias("prev_adj_close"))
    mkt = d.group_by("date").agg(pl.col("ret").median().alias("mkt_med"),
                                 (pl.col("ret").abs() > 0.10).sum().alias("n_big10"))
    d = d.join(mkt, on="date", how="left")
    cls = (pl.when(~pl.col("review")).then(pl.lit(None, dtype=pl.Utf8))
           .when(pl.col("manual_event")).then(pl.lit("demerger_unadjusted"))
           .when(pl.col("prev_adj_close") < 20).then(pl.lit("tick_quantization"))
           .when((pl.col("mkt_med").abs() > 0.07) | (pl.col("n_big10") >= 8))
           .then(pl.lit("market_wide"))
           .when(pl.col("n_big10") >= 3).then(pl.lit("sector_group"))
           .when(pl.col("ret").abs() > 0.60).then(pl.lit("suspected_unadjusted"))
           .when(pl.col("vol_x") >= 2.5).then(pl.lit("news_vol_spike"))
           .otherwise(pl.lit("unclear")))
    d = d.with_columns(cls.alias("ret_class"))
    return d.with_columns(pl.col("ret_class").is_in(EXCLUDE).fill_null(False).alias("exclude_ret")) \
            .drop("vol_x", "prev_adj_close", "mkt_med", "n_big10")


def main():
    files = sorted(glob.glob(str(DATA / "clean" / "adj_daily" / "*.parquet")))
    d = pl.concat([pl.read_parquet(f).with_columns(pl.lit(Path(f).stem).alias("sym")) for f in files])
    d = classify(d)
    for (s,), g in d.group_by("sym"):
        g.drop("sym").sort("date").write_parquet(DATA / "clean" / "adj_daily" / f"{s}.parquet")
    r = d.filter(pl.col("review"))
    print(r.group_by("ret_class").len().sort("len", descending=True))
    print("exclude_ret rows:", d["exclude_ret"].sum())
    r.select(["sym", "date", "ret", "ret_class"]).sort("ret_class", "date") \
        .write_csv(DATA / "clean" / "review_classified.csv")


if __name__ == "__main__":
    main()
