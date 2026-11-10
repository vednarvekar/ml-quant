"""Cross-check Upstox adjusted daily prices against NSE raw (unadjusted) bhavcopy.

FINDING (2026-11-10): NSE bhavcopy PREV_CLOSE is NOT adjusted on ex-dates, so it cannot be
chained into an adjusted series (an early attempt produced fake -32% on IOC's 1:2 bonus).
Upstox daily IS adjusted (splits, bonus, apparently dividends; e.g. Bajaj Finance raw -89.9%
-> +0.5%). So: research basis = Upstox daily (`close`), NSE raw kept alongside:

    ret      = Upstox adjusted daily return
    ret_raw  = close_raw / prev_close_raw - 1          (NSE, unadjusted)
    adj_step = |ret - ret_raw| > 0.3%   -> ex-date adjustment step (split/bonus/dividend)
    review   = |ret| > 25%              -> real crash/jump OR unadjusted demerger; see MANUAL

Entities (renames, ISIN changes) are linked through (symbol, ISIN) components.

uv run python -m mlq.adjust
"""
import glob
from pathlib import Path

import polars as pl

from mlq.config import DATA

BHAV = DATA / "raw" / "nse_bhav"
OUT = DATA / "clean" / "adj_daily"
# new symbol -> former symbol (renames after NSE dropped ISIN from the bhavcopy format)
ALIASES = {"ETERNAL": "ZOMATO", "LTM": "LTIM", "TMPV": "TATAMOTORS"}
# Large drops Upstox did NOT adjust that look like demergers (UNVERIFIED, from price pattern).
MANUAL = [("SIEMENS", "2025-04-07"), ("TMPV", "2025-10-14"), ("VEDL", "2026-04-30")]


def load_bhav() -> pl.DataFrame:
    fs = sorted(glob.glob(str(BHAV / "*.parquet")))
    return pl.concat([
        pl.read_parquet(f).with_columns(pl.lit(Path(f).stem).str.to_date().alias("date"))
        for f in fs])


def components(pairs: list[tuple[str, str]]) -> dict[str, int]:
    """Union-find over symbols and ISINs; returns node -> component id."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, i in pairs:
        parent[find("S:" + s)] = find("I:" + i)
    roots: dict[str, int] = {}
    return {n: roots.setdefault(find(n), len(roots)) for n in list(parent)}


def raw_series(b: pl.DataFrame, current_symbols: list[str]) -> pl.DataFrame:
    """NSE raw (unadjusted) close/prev_close per current symbol, linked across renames."""
    pairs = b.filter(pl.col("ISIN").is_not_null()).select(["nse_symbol", "ISIN"]).unique()
    comp = components([(s, i) for s, i in pairs.iter_rows()])
    sym_comp = {n[2:]: c for n, c in comp.items() if n.startswith("S:")}
    for new, old in ALIASES.items():
        if old in sym_comp:
            sym_comp[new] = sym_comp[old]
    seen = set(b["nse_symbol"].unique().to_list())
    nxt = max(sym_comp.values(), default=0) + 1
    for s in current_symbols:
        if s not in sym_comp and s in seen:
            sym_comp[s] = nxt
            nxt += 1
    cid_to_sym = {sym_comp[s]: s for s in current_symbols if s in sym_comp}
    b = b.with_columns(pl.col("nse_symbol").replace_strict(sym_comp, default=None).alias("cid"))
    b = b.filter(pl.col("cid").is_in(list(cid_to_sym))) \
         .with_columns(pl.col("cid").replace_strict(cid_to_sym).alias("sym"))
    b = b.sort(["sym", "date", "volume"]).unique(["sym", "date"], keep="last")
    b = b.filter((pl.col("close") > 0) & (pl.col("prev_close") > 0))
    return b.select(["sym", "date", pl.col("close").alias("raw_close"),
                     pl.col("prev_close").alias("raw_prev_close")]).sort(["sym", "date"])


def combine(up: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """up: sym,date,close (Upstox adjusted daily). Returns research daily table with flags."""
    d = up.join(raw, on=["sym", "date"], how="left").sort(["sym", "date"])
    d = d.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("sym") - 1).alias("ret"),
        (pl.col("raw_close") / pl.col("raw_prev_close") - 1).alias("ret_raw"))
    d = d.with_columns(
        ((pl.col("ret") - pl.col("ret_raw")).abs() > 0.003).fill_null(False).alias("adj_step"),
        (pl.col("ret").abs() > 0.25).fill_null(False).alias("review"))
    manual = pl.DataFrame({"sym": [m[0] for m in MANUAL],
                           "date": [pl.Series([m[1]]).str.to_date()[0] for m in MANUAL],
                           "manual_event": [True] * len(MANUAL)})
    return d.join(manual, on=["sym", "date"], how="left").with_columns(
        pl.col("manual_event").fill_null(False))


def main():
    syms = sorted(Path(f).stem for f in glob.glob(str(DATA / "clean" / "1d" / "*.parquet")))
    raw = raw_series(load_bhav(), syms)
    up = pl.concat([pl.read_parquet(DATA / "clean" / "1d" / f"{s}.parquet") for s in syms]) \
        .select(["sym", "date", "close", "open", "high", "low", "volume"])
    d = combine(up, raw)
    OUT.mkdir(parents=True, exist_ok=True)
    for (s,), g in d.group_by("sym"):
        g.drop("sym").sort("date").write_parquet(OUT / f"{s}.parquet")
    nr = d.filter(pl.col("raw_close").is_null()).height
    print(f"symbols {d['sym'].n_unique()} rows {d.height}; no NSE raw match: {nr}")
    print("adj steps:", d["adj_step"].sum(), "| review (>25% moves):", d["review"].sum(),
          "| manual events:", d["manual_event"].sum())


if __name__ == "__main__":
    main()
