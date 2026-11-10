"""Trading calendar + missing-day check (reads clean data only; no network).

Calendar = dates where >=30 symbols traded in Upstox daily (same rule as nse_bhav.trading_days).
For each symbol: expected days = calendar days between its first and last date;
missing = expected - present. Writes data/clean/missing_days.csv.

uv run python -m mlq.calendar_check
"""
import polars as pl

from mlq.config import DATA

CLEAN = DATA / "clean"


def calendar(d: pl.DataFrame, min_syms: int = 30) -> list:
    return (d.group_by("date").agg(pl.len().alias("n")).filter(pl.col("n") >= min_syms)
            .sort("date")["date"].to_list())


def missing_per_symbol(d: pl.DataFrame, cal: list) -> pl.DataFrame:
    cal_df = pl.DataFrame({"date": cal})
    rows = []
    for (sym,), g in d.group_by("sym"):
        lo, hi = g["date"].min(), g["date"].max()
        exp = cal_df.filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
        miss = exp.join(g.select("date"), on="date", how="anti")["date"].to_list()
        rows.append({"sym": sym, "first": lo, "last": hi, "expected": exp.height,
                     "missing": len(miss), "missing_dates": ",".join(map(str, miss[:10]))})
    return pl.DataFrame(rows).sort("missing", descending=True)


def main():
    d = pl.read_parquet(CLEAN / "1d" / "*.parquet")
    cal = calendar(d)
    print(f"calendar: {len(cal)} trading days {cal[0]}..{cal[-1]}")
    per_year = pl.DataFrame({"date": cal}).group_by(pl.col("date").dt.year().alias("year")) \
        .agg(pl.len().alias("days")).sort("year")
    print(per_year)
    m = missing_per_symbol(d, cal)
    m.write_csv(CLEAN / "missing_days.csv")
    print("symbols with any missing day:", m.filter(pl.col("missing") > 0).height, "of", m.height)
    print(m.head(15))


if __name__ == "__main__":
    main()
