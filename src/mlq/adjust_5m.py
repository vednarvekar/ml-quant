"""Put Upstox 5m bars on the same basis as Upstox daily (our research basis).

Upstox 5m and daily use slightly different adjustment factors (e.g. IOC 2022: 5m ~0.9x the
daily basis). Per day, rho_t = last5m_close_t / daily_close_t estimates the ratio but is noisy
(last candle != official close). Between breakpoints (ex-date adjustment steps from
adjust.py and >1% jumps in rho) the true ratio is constant, so we divide the bars by the
segment median of rho.

uv run python -m mlq.adjust_5m
"""
import glob
from pathlib import Path

import polars as pl

from mlq.config import DATA

OUT = DATA / "clean" / "adj_5m"


def scale_table(m: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    last = (m.sort("ts").group_by("date").agg(pl.col("close").last().alias("c5"),
                                              pl.len().alias("bars"))
            .filter(pl.col("bars") >= 70).sort("date"))
    t = last.join(daily.select(["date", "close", "adj_step"]), on="date", how="inner").sort("date")
    t = t.with_columns((pl.col("c5") / pl.col("close")).alias("rho"))
    t = t.with_columns(
        (pl.col("adj_step")
         | ((pl.col("rho") / pl.col("rho").shift(1) - 1).abs() > 0.01)).fill_null(False).alias("brk"))
    t = t.with_columns(pl.col("brk").cast(pl.Int32).cum_sum().alias("seg"))
    t = t.with_columns(pl.col("rho").median().over("seg").alias("rho_seg"),
                       pl.len().over("seg").alias("seg_len"))
    return t.select(["date", "rho_seg", "seg_len"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(str(DATA / "clean" / "5m" / "*.parquet"))):
        sym = Path(f).stem
        m = pl.read_parquet(f)
        daily = pl.read_parquet(DATA / "clean" / "adj_daily" / f"{sym}.parquet")
        st = scale_table(m, daily)
        x = m.join(st, on="date", how="left")
        x = x.with_columns([(pl.col(c) / pl.col("rho_seg")).alias(c)
                            for c in ("open", "high", "low", "close")])
        x = x.filter(pl.col("rho_seg").is_not_null())
        x = x.drop("rho_seg", "seg_len")
        x.write_parquet(OUT / f"{sym}.parquet")
        v = x.filter(~pl.col("special_session")).sort("ts").group_by("date").agg(
            pl.col("close").last().alias("c5"), pl.len().alias("bars")).filter(pl.col("bars") == 75)
        v = v.join(daily.select(["date", "close"]), on="date")
        gap = (v["c5"] / v["close"] - 1).abs()
        rows.append({"sym": sym, "days": v.height, "gap_gt0.5pct": (gap > 0.005).mean(),
                     "gap_gt2pct": (gap > 0.02).mean(), "dropped_rows": m.height - x.height,
                     "short_segments": st.filter(pl.col("seg_len") < 5).height})
    r = pl.DataFrame(rows).sort("gap_gt0.5pct", descending=True)
    r.write_csv(DATA / "clean" / "adj_5m_validation.csv")
    print(r.select(pl.col("gap_gt0.5pct").mean().alias("avg gap>0.5%"),
                   pl.col("gap_gt2pct").mean().alias("avg gap>2%"),
                   pl.col("dropped_rows").sum()).to_dicts())
    print(r.head(8))


if __name__ == "__main__":
    main()
