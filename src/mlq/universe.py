"""Point-in-time liquid universe (fixes survivorship bias).

Problem: our stock list is TODAY's Nifty 100, so the past only contains companies that
later did well. Fix: for every month, rank ALL NSE stocks (incl. ones later delisted) by
their 60-day median traded value (close * volume) using the bhavcopy files, and keep the
top N. A backtest on date D may only trade stocks in the universe of D's month.

Proxy note: real Nifty 100 uses free-float market cap; liquidity rank is a close, fully
reproducible stand-in and is what matters for tradability anyway.

uv run python -m mlq.universe
"""
import polars as pl

from mlq.adjust import components, load_bhav
from mlq.config import DATA, INSTRUMENTS_FILE

OUT = DATA / "clean" / "universe.parquet"


def entity_map(b: pl.DataFrame) -> pl.DataFrame:
    """nse_symbol -> entity id (renames/ISIN changes linked) + latest symbol name."""
    pairs = b.filter(pl.col("ISIN").is_not_null()).select(["nse_symbol", "ISIN"]).unique()
    comp = components([(s, i) for s, i in pairs.iter_rows()])
    sym_comp = {n[2:]: c for n, c in comp.items() if n.startswith("S:")}
    nxt = max(sym_comp.values(), default=0) + 1
    for s in b["nse_symbol"].unique().to_list():  # symbols never seen with ISIN
        if s not in sym_comp:
            sym_comp[s] = nxt
            nxt += 1
    m = pl.DataFrame({"nse_symbol": list(sym_comp), "entity": list(sym_comp.values())})
    last = (b.group_by("nse_symbol").agg(pl.col("date").max().alias("last_seen"))
            .join(m, on="nse_symbol").sort("last_seen")
            .group_by("entity").agg(pl.col("nse_symbol").last().alias("name"),
                                    pl.col("last_seen").max()))
    return m.join(last, on="entity")


def fund_symbols(b: pl.DataFrame) -> set[str]:
    """ETFs / fund units trade as 'EQ' too; their ISINs start with INF (companies: INE)."""
    s = set(b.filter(pl.col("ISIN").str.starts_with("INF"))["nse_symbol"].unique().to_list())
    if INSTRUMENTS_FILE.exists():
        i = pl.read_parquet(INSTRUMENTS_FILE).filter(pl.col("segment") == "NSE_EQ")
        s |= set(i.filter(pl.col("isin").str.starts_with("INF"))["trading_symbol"].to_list())
    return s


def build(b: pl.DataFrame, top_n: int = 200, window: int = 60) -> pl.DataFrame:
    b = b.filter(~pl.col("nse_symbol").is_in(list(fund_symbols(b))))
    em = entity_map(b)
    v = (b.join(em.select(["nse_symbol", "entity", "name", "last_seen"]), on="nse_symbol")
         .with_columns((pl.col("close") * pl.col("volume")).alias("value"))
         .group_by(["entity", "date"]).agg(pl.col("value").sum(), pl.col("name").first(),
                                           pl.col("last_seen").first())
         .sort(["entity", "date"]))
    v = v.with_columns(
        pl.col("value").rolling_median(window, min_samples=40).over("entity").alias("med_value"),
        pl.col("date").dt.truncate("1mo").alias("month"))
    snap = (v.filter(pl.col("med_value").is_not_null())
            .group_by(["entity", "month"]).agg(pl.all().sort_by("date").last()))
    snap = snap.with_columns(
        pl.col("med_value").rank("ordinal", descending=True).over("month").alias("rank"))
    return snap.filter(pl.col("rank") <= top_n).select(
        ["month", "entity", "name", "rank", "med_value", "last_seen"]).sort(["month", "rank"])


def main():
    u = build(load_bhav())
    u.write_parquet(OUT)
    current = set(pl.read_csv(DATA / "nifty100_list.csv")["Symbol"].to_list())
    listed = set(pl.read_parquet(INSTRUMENTS_FILE)
                 .filter((pl.col("segment") == "NSE_EQ") & (pl.col("instrument_type") == "EQ"))
                 ["trading_symbol"].to_list())
    for since, n in (("2005-01-01", 100), ("2022-01-01", 100), ("2022-01-01", 200)):
        ever = (u.filter((pl.col("month") >= pl.lit(since).str.to_date()) & (pl.col("rank") <= n))
                .select(["entity", "name", "last_seen"]).unique("entity"))
        extra = ever.filter(~pl.col("name").is_in(list(current)))
        still = extra.filter(pl.col("name").is_in(list(listed)))
        print(f"top{n} since {since}: {ever.height} companies ever in; {extra.height} not in "
              f"today's Nifty 100; of those {still.height} still listed (Upstox has data), "
              f"{extra.height - still.height} delisted/renamed (no Upstox history)")
    print("months:", u["month"].n_unique(), "| file:", OUT)


if __name__ == "__main__":
    main()
