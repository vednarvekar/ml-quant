"""Phase 3 baselines on the daily engine (delivery, long-only, monthly rebalance).

Universe is point-in-time (liquidity top-N from universe.parquet, month of the signal date, i.e.
older information than needed). Only symbols we hold Upstox data for can be used, so coverage
is printed; from 2022 it is high (we downloaded every 2022+ top-100 name still listed).

uv run python -m mlq.baselines
"""
import glob
from pathlib import Path

import numpy as np
import polars as pl

from mlq.backtest import panel, simulate
from mlq.config import DATA

START = "2022-01-03"


def load_prices() -> pl.DataFrame:
    fs = glob.glob(str(DATA / "clean" / "adj_daily" / "*.parquet"))
    return pl.concat([pl.read_parquet(f).with_columns(pl.lit(Path(f).stem).alias("sym"))
                      .select(["sym", "date", "open", "high", "low", "close", "volume", "exclude_ret"]) for f in fs])


def month_end_idx(dates: list) -> list[int]:
    return [i for i in range(len(dates) - 1)
            if (dates[i].year, dates[i].month) != (dates[i + 1].year, dates[i + 1].month)]


def build_weights(dates, syms, close, uni: pl.DataFrame, top_n: int, mode: str,
                  pick: int = 20) -> tuple[np.ndarray, list[float]]:
    T, N = close.shape
    w = np.zeros((T, N))
    col = {s: j for j, s in enumerate(syms)}
    cover, cur = [], np.zeros(N)
    me = set(month_end_idx(dates))
    for t in range(T):
        if t in me and dates[t].isoformat() >= START:
            m = dates[t].replace(day=1)
            names = uni.filter((pl.col("month") == m) & (pl.col("rank") <= top_n))["name"].to_list()
            have = [col[n] for n in names if n in col and not np.isnan(close[t, col[n]])]
            cover.append(len(have) / max(len(names), 1))
            cur = np.zeros(N)
            if mode == "momentum" and t >= 252 and have:
                sc = {j: close[t - 21, j] / close[t - 252, j] - 1 for j in have
                      if not np.isnan(close[t - 21, j]) and not np.isnan(close[t - 252, j])}
                have = sorted(sc, key=sc.get, reverse=True)[:pick]
            if have:
                cur[have] = 1.0 / len(have)
        w[t] = cur
    return w, cover


def main():
    p = load_prices()
    dates, syms, close = panel(p, "close")
    _, _, open_ = panel(p, "open")
    uni = pl.read_parquet(DATA / "clean" / "universe.parquet")
    i0 = next(i for i, d in enumerate(dates) if d.isoformat() >= START)
    i0 = max(i0 - 1, 0)
    rows = []
    for name, top_n, mode in [("EW top-50", 50, "ew"), ("EW top-100", 100, "ew"),
                              ("Momentum 12-1 (20 of top-100)", 100, "momentum")]:
        w, cover = build_weights(dates, syms, close, uni, top_n, mode)
        r = simulate(open_[i0:], close[i0:], w[i0:], dates[i0:])
        m = r.metrics()
        rows.append({"strategy": name, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "max_dd": m["max_dd"], "turnover/yr": m["turnover_per_year"],
                     "fees_rs": m["fees"], "coverage": float(np.mean(cover))})
    # benchmark: Nifty 50 index buy & hold (no costs)
    idx = pl.read_parquet(DATA / "clean" / "idx" / "1d" / "Nifty_50.parquet") \
        .filter(pl.col("date") >= pl.lit(START).str.to_date()).sort("date")
    c = idx["close"].to_numpy()
    r = c[1:] / c[:-1] - 1
    yrs = len(r) / 252
    rows.append({"strategy": "Nifty 50 buy&hold (index, no costs)", "cagr": (c[-1] / c[0]) ** (1 / yrs) - 1,
                 "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(252),
                 "max_dd": float((c / np.maximum.accumulate(c) - 1).min()),
                 "turnover/yr": 0.0, "fees_rs": 0.0, "coverage": 1.0})
    out = pl.DataFrame(rows)
    print(out)
    out.write_csv(DATA / "clean" / "baselines.csv")


if __name__ == "__main__":
    main()
