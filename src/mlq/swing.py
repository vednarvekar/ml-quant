"""Swing (daily) strategy families run on the honest daily engine.

Each family is a function (ctx, t) -> target weight vector using data up to the close of day t.
Rebalance schedule: month-end or week-end. Every run is logged in the experiment registry.

Windows:
  clean: 2022-01 -> now      (point-in-time universe has high data coverage)
  long : 2012-01 -> 2021-12  (survivor-biased: delisted names missing; coverage printed)

uv run python -m mlq.swing
"""
from dataclasses import dataclass

import numpy as np
import polars as pl

from mlq import registry
from mlq.backtest import panel, simulate
from mlq.baselines import load_prices, month_end_idx
from mlq.config import DATA
from mlq.futures import lot_sizes

WINDOWS = {"clean": ("2022-01-03", "2026-12-31"), "long": ("2012-01-02", "2021-12-31")}


@dataclass
class Ctx:
    dates: list
    syms: list
    close: np.ndarray
    uni: pl.DataFrame
    idx_close: np.ndarray          # Nifty 50 aligned to dates (NaN if missing)
    fno: np.ndarray                # bool N: has stock futures today
    _cache: dict

    def members(self, t: int, top_n: int) -> np.ndarray:
        m = self.dates[t].replace(day=1)
        key = (m, top_n)
        if key not in self._cache:
            names = set(self.uni.filter((pl.col("month") == m) & (pl.col("rank") <= top_n))
                        ["name"].to_list())
            self._cache[key] = (np.array([s in names for s in self.syms]),
                                len(names))
        return self._cache[key][0]

    def coverage(self, t: int, top_n: int) -> float:
        self.members(t, top_n)
        have = self._cache[(self.dates[t].replace(day=1), top_n)]
        return have[0].sum() / max(have[1], 1)


def week_end_idx(dates: list) -> list[int]:
    return [i for i in range(len(dates) - 1) if dates[i].isocalendar()[:2] != dates[i + 1].isocalendar()[:2]]


def valid(ctx, t, mask, lookback):
    ok = mask & ~np.isnan(ctx.close[t])
    if t - lookback >= 0:
        ok &= ~np.isnan(ctx.close[t - lookback])
    return ok


def trend_on(ctx, t, n=200) -> bool:
    if t < n or np.isnan(ctx.idx_close[t]):
        return True
    w = ctx.idx_close[t - n + 1:t + 1]
    return bool(ctx.idx_close[t] > np.nanmean(w))


# ---- families -------------------------------------------------------------------------------
def f_ew(top_n):
    def f(ctx, t, held):
        ok = valid(ctx, t, ctx.members(t, top_n), 0)
        w = np.zeros(len(ctx.syms))
        if ok.any():
            w[ok] = 1 / ok.sum()
        return w
    return f


def f_trend(top_n, n=200):
    base = f_ew(top_n)
    return lambda ctx, t, held: base(ctx, t, held) * (1.0 if trend_on(ctx, t, n) else 0.0)


def f_mom(top_n, pick, keep=None, lb=252, skip=21):
    def f(ctx, t, held):
        ok = valid(ctx, t, ctx.members(t, top_n), lb)
        ok &= ~np.isnan(ctx.close[t - skip]) if t >= skip else ok
        w = np.zeros(len(ctx.syms))
        if t < lb or not ok.any():
            return w
        sc = np.full(len(ctx.syms), -np.inf)
        sc[ok] = ctx.close[t - skip, ok] / ctx.close[t - lb, ok] - 1
        order = [j for j in np.argsort(-sc) if np.isfinite(sc[j])]
        rank = {j: r for r, j in enumerate(order)}
        chosen = []
        if keep:                                    # hysteresis: keep holdings still within top `keep`
            chosen = sorted((j for j in np.nonzero(held > 0)[0] if rank.get(j, 10**9) < keep),
                            key=lambda j: rank[j])[:pick]
        for j in order:
            if len(chosen) >= pick:
                break
            if j not in chosen:
                chosen.append(j)
        for j in chosen:
            w[j] = 1 / len(chosen)
        return w
    return f


def f_lowvol(top_n, pick, lb=60):
    def f(ctx, t, held):
        ok = valid(ctx, t, ctx.members(t, top_n), lb)
        w = np.zeros(len(ctx.syms))
        if t < lb or not ok.any():
            return w
        r = ctx.close[t - lb + 1:t + 1] / ctx.close[t - lb:t] - 1
        vol = np.where(ok, np.nanstd(r, axis=0), np.inf)
        ch = np.argsort(vol)[:pick]
        ch = ch[np.isfinite(vol[ch])]
        w[ch] = 1 / len(ch) if len(ch) else 0
        return w
    return f


def f_reversal(top_n, pick, lb=5):
    def f(ctx, t, held):
        ok = valid(ctx, t, ctx.members(t, top_n), lb)
        w = np.zeros(len(ctx.syms))
        if t < lb or not ok.any():
            return w
        sc = np.where(ok, ctx.close[t] / ctx.close[t - lb] - 1, np.inf)
        ch = np.argsort(sc)[:pick]
        ch = ch[np.isfinite(sc[ch])]
        w[ch] = 1 / len(ch) if len(ch) else 0
        return w
    return f


def f_longshort(top_n, pick, lb=252, skip=21):
    """Futures: long top momentum / short bottom momentum among F&O names, gross 1."""
    def f(ctx, t, held):
        ok = valid(ctx, t, ctx.members(t, top_n) & ctx.fno, lb)
        w = np.zeros(len(ctx.syms))
        if t < lb or ok.sum() < 2 * pick:
            return w
        sc = np.where(ok, ctx.close[t - skip] / ctx.close[t - lb] - 1, np.nan)
        order = np.argsort(np.where(np.isnan(sc), -np.inf, sc))   # ascending, nan first
        valid_idx = [j for j in order if not np.isnan(sc[j])]
        for j in valid_idx[-pick:]:
            w[j] = 0.5 / pick
        for j in valid_idx[:pick]:
            w[j] = -0.5 / pick
        return w
    return f


def f_combo(top_n, pick):
    mom = f_mom(top_n, pick * 2)
    lv = f_lowvol(top_n, pick * 2)

    def f(ctx, t, held):
        a, b = mom(ctx, t, held), lv(ctx, t, held)
        both = (a > 0) & (b > 0)
        w = np.zeros(len(ctx.syms))
        if both.sum() >= 5:
            w[both] = 1 / both.sum()
        else:
            w = 0.5 * a + 0.5 * b
        return w * (1.0 if trend_on(ctx, t) else 0.5)
    return f


FAMILIES = [  # name, factory, schedule, kind
    ("EW top-100 monthly", lambda: f_ew(100), "m", "delivery"),
    ("Trend200 + EW top-100", lambda: f_trend(100), "m", "delivery"),
    ("Mom 12-1 top20/100", lambda: f_mom(100, 20), "m", "delivery"),
    ("Mom 12-1 top20 keep40", lambda: f_mom(100, 20, keep=40), "m", "delivery"),
    ("LowVol60 20/100", lambda: f_lowvol(100, 20), "m", "delivery"),
    ("Reversal5d 10/100 weekly", lambda: f_reversal(100, 10), "w", "delivery"),
    ("Combo mom+lowvol+trend", lambda: f_combo(100, 10), "m", "delivery"),
    ("L/S mom F&O 10/10 (futures)", lambda: f_longshort(150, 10), "m", "futures"),
]


def weights_for(ctx, fn, sched, i_start):
    T, N = ctx.close.shape
    reb = set(month_end_idx(ctx.dates) if sched == "m" else week_end_idx(ctx.dates))
    w = np.zeros((T, N))
    cur = np.zeros(N)
    for t in range(max(i_start - 1, 0), T):
        if t in reb:
            cur = fn(ctx, t, cur)
        w[t] = cur
    return w


def make_ctx() -> Ctx:
    p = load_prices()
    dates, syms, close = panel(p, "close")
    idx = pl.read_parquet(DATA / "clean" / "idx" / "1d" / "Nifty_50.parquet").select(["date", "close"])
    m = dict(zip(idx["date"].to_list(), idx["close"].to_list()))
    ic = np.array([m.get(d, np.nan) for d in dates], float)
    lots = lot_sizes()
    return Ctx(dates, syms, close, pl.read_parquet(DATA / "clean" / "universe.parquet"), ic,
               np.array([s in lots for s in syms]), {})


def run_window(ctx, open_, wname, rows):
    lo, hi = WINDOWS[wname]
    i0 = next(i for i, d in enumerate(ctx.dates) if d.isoformat() >= lo)
    i1 = max(i for i, d in enumerate(ctx.dates) if d.isoformat() <= hi) + 1
    cov = np.mean([ctx.coverage(t, 100) for t in month_end_idx(ctx.dates) if i0 <= t < i1])
    print(f"[{wname}] {ctx.dates[i0]}..{ctx.dates[i1-1]} pit coverage {cov:.0%}", flush=True)
    for name, fac, sched, kind in FAMILIES:
        w = weights_for(ctx, fac(), sched, i0)
        sl = slice(max(i0 - 1, 0), i1)
        r = simulate(open_[sl], ctx.close[sl], w[sl], ctx.dates[sl], kind=kind,
                     max_gross=1.0 + 1e-9)
        m = r.metrics()
        e = r.equity["equity"].to_numpy()
        daily = e[1:] / e[:-1] - 1
        registry.log_trial(f"{name} [{wname}]", "swing", {"window": wname, "kind": kind}, daily,
                           {"cagr": m["cagr"], "max_dd": m["max_dd"]})
        rows.append({"window": wname, "strategy": name, "cagr": round(m["cagr"], 4),
                     "sharpe": round(m["sharpe"], 2), "max_dd": round(m["max_dd"], 3),
                     "turn/yr": round(m["turnover_per_year"], 1), "fees_k": round(m["fees"] / 1e3),
                     "cov": round(cov, 2), "_daily": daily})


def main():
    ctx = make_ctx()
    _, _, open_ = panel(load_prices(), "open")
    rows = []
    for wn in WINDOWS:
        run_window(ctx, open_, wn, rows)
    out = pl.DataFrame([{k: v for k, v in r.items() if k != "_daily"} for r in rows])
    with pl.Config(tbl_rows=40, tbl_width_chars=160, fmt_str_lengths=40):
        print(out)
    out.write_csv(DATA / "clean" / "swing_results.csv")
    # deflated sharpe of each vs all swing trials
    print("\nDSR (prob true Sharpe>0 after", registry.n_trials("swing"), "swing trials):")
    for r in rows:
        print(f"  {r['window']:5} {r['strategy']:32} DSR={registry.dsr(r['_daily'], 'swing'):.2f}")


if __name__ == "__main__":
    main()
