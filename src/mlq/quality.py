"""Data quality report for raw Upstox candles.

uv run python -m mlq.quality   -> prints summary, writes data/quality_report.md
"""
import glob
from pathlib import Path

import polars as pl

from mlq.config import RAW, DATA


def load_all(tf: str) -> pl.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(RAW / tf / "*.parquet"))):
        frames.append(pl.read_parquet(f).with_columns(pl.lit(Path(f).stem).alias("sym")))
    return pl.concat(frames)


def ohlc_violations(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        (pl.col("high") < pl.max_horizontal("open", "close", "low"))
        | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
        | (pl.col("close") <= 0) | (pl.col("open") <= 0)
        | (pl.col("high") <= 0) | (pl.col("low") <= 0)
    )


def daily_checks(d: pl.DataFrame) -> dict:
    d = d.with_columns(pl.col("ts").dt.date().alias("date"))
    out = {"rows": d.height, "symbols": d["sym"].n_unique()}
    out["dup_rows"] = d.height - d.unique(["sym", "date"]).height
    out["ohlc_violations"] = ohlc_violations(d).height
    d = d.sort(["sym", "date"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("sym") - 1).alias("ret"))
    big = d.filter(pl.col("ret").abs() > 0.2).select(["sym", "date", "close", "ret"])
    out["big_moves_gt20pct"] = big
    out["zero_volume_days"] = d.filter(pl.col("volume") == 0).height
    cov = d.group_by("sym").agg(pl.col("date").min().alias("first"), pl.len().alias("n"))
    out["coverage"] = cov.sort("first", descending=True)
    # calendar: dates traded by >=50% of symbols (post 2022) vs per-symbol gaps
    cal = (d.group_by("date").agg(pl.len().alias("n")).filter(pl.col("n") >= 30)["date"])
    out["calendar_days"] = cal.len()
    return out


def intraday_checks(m: pl.DataFrame, d: pl.DataFrame) -> dict:
    m = m.with_columns(pl.col("ts").dt.date().alias("date"))
    out = {"rows": m.height}
    out["dup_rows"] = m.height - m.unique(["sym", "ts"]).height
    out["ohlc_violations"] = ohlc_violations(m).height
    out["zero_volume_bars"] = m.filter(pl.col("volume") == 0).height
    per_day = m.group_by(["sym", "date"]).agg(pl.len().alias("bars"))
    out["bars_per_day_dist"] = per_day.group_by("bars").len().sort("len", descending=True).head(8)
    out["short_days"] = per_day.filter(pl.col("bars") < 70).height
    # consistency: sum of 5m volume vs daily volume, last 5m close vs daily close
    agg = m.sort("ts").group_by(["sym", "date"]).agg(
        pl.col("volume").sum().alias("v5"), pl.col("close").last().alias("c5"))
    dd = d.with_columns(pl.col("ts").dt.date().alias("date")).select(
        ["sym", "date", pl.col("volume").alias("vd"), pl.col("close").alias("cd")])
    j = agg.join(dd, on=["sym", "date"], how="inner")
    out["joined_days"] = j.height
    out["close_mismatch_gt0.5pct"] = j.filter(((pl.col("c5") / pl.col("cd")) - 1).abs() > 0.005).height
    out["volume_mismatch_gt5pct"] = j.filter(
        (pl.col("vd") > 0) & (((pl.col("v5") / pl.col("vd")) - 1).abs() > 0.05)).height
    out["days_5m_missing_in_daily"] = agg.join(dd, on=["sym", "date"], how="anti").height
    return out


def main():
    d, m = load_all("1d"), load_all("5m")
    dc, ic = daily_checks(d), intraday_checks(m, d)
    lines = ["# Data quality report (Upstox raw, Nifty 100)\n", "## Daily"]
    for k, v in dc.items():
        if isinstance(v, pl.DataFrame):
            lines += [f"\n**{k}** ({v.height} rows)\n", "```", str(v.head(25)), "```"]
        else:
            lines.append(f"- {k}: {v}")
    lines.append("\n## 5-minute")
    for k, v in ic.items():
        if isinstance(v, pl.DataFrame):
            lines += [f"\n**{k}**\n", "```", str(v), "```"]
        else:
            lines.append(f"- {k}: {v}")
    text = "\n".join(lines)
    (DATA / "quality_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
