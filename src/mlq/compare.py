"""Phase-averaged comparison of the ML ranker against fixed baselines (audit step 1).

uv run python -m mlq.compare [pred_file]
Default pred file: data/clean/ml_swing_pred_h5.npy
"""
import sys

import numpy as np
import polars as pl

from mlq.backtest import panel, simulate
from mlq.baselines import load_prices
from mlq.config import DATA
from mlq.evaluate import CLEAN_START, idx_returns, index_rows, run_phases
from mlq.futures import lot_sizes
from mlq.futures_bt import simulate_futures
from mlq.swing import make_ctx
from mlq.swing_portfolio import build_weights


def ew_weights(pred, first, R):
    T, N = pred.shape
    W = np.zeros((T, N))
    cur, since = np.zeros(N), 10**9
    for t in range(max(first - 1, 0), T):
        since += 1
        if since >= R and np.isfinite(pred[t]).sum() >= 5:
            since = 0
            ok = np.isfinite(pred[t])
            cur = np.zeros(N)
            cur[ok] = 1.0 / ok.sum()
        W[t] = cur
    return W


def main(pred_file=None):
    pred0 = np.load(pred_file or DATA / "clean" / "ml_swing_pred_h5.npy")
    p = load_prices()
    ctx = make_ctx()
    _, _, O = panel(p, "open")
    T, N = ctx.close.shape
    i0 = next(i for i, d in enumerate(ctx.dates) if d.isoformat() >= CLEAN_START)
    fac = {k: v[i0:] for k, v in idx_returns(ctx.dates).items()}
    lots = lot_sizes()
    lot_arr = np.array([lots.get(s, np.nan) for s in ctx.syms], float)
    sl = slice(i0 - 1, T)
    all_rows = []
    for mode, kw in (("delivery long-only, 10L, top-20", dict(k=20, keep=40, kind="delivery", cap=1_000_000)),
                     ("futures long-only, 20L, top-10", dict(k=10, keep=20, kind="futures", cap=2_000_000))):
        pred = np.where(ctx.fno[None, :], pred0, np.nan) if kw["kind"] == "futures" else pred0
        R = 5

        def returns_from(W):
            if kw["kind"] == "delivery":
                r = simulate(O[sl], ctx.close[sl], W[sl], ctx.dates[sl], kind="delivery", slippage_bps=5.0)
                e = r.equity["equity"].to_numpy()
            else:
                e = simulate_futures(O[sl], ctx.close[sl], W[sl], ctx.dates[sl], lot_arr,
                                     capital=kw["cap"], slippage_bps=5.0).equity
            return e[1:] / e[:-1] - 1

        def scored(score_fn, name):
            def runner(ph):
                sc = score_fn()
                return returns_from(build_weights(sc, ctx, i0 - ph, R, "hyst", k=kw["k"], keep=kw["keep"]))
            return runner

        def rev_scores():
            s = np.full_like(pred, np.nan)
            s[5:] = -(ctx.close[5:] / ctx.close[:-5] - 1)
            return np.where(np.isfinite(pred), s, np.nan)

        runners = {
            "ML ranker": scored(lambda: pred, "ml"),
            "5-day reversal rule": scored(rev_scores, "rev"),
            "universe equal-weight (no model)": lambda ph: returns_from(ew_weights(pred, i0 - ph, R)),
        }
        rows = []
        for name, rn in runners.items():
            rows.append(run_phases(name, rn, list(range(R)), fac).summary())
        rnd = []
        for seed in range(6):
            rng = np.random.default_rng(seed)
            rs = np.where(np.isfinite(pred), rng.normal(size=pred.shape), np.nan)
            rnd.append(run_phases(f"rand{seed}", scored(lambda rs=rs: rs, "r"), [0, 2, 4], fac))
        sh = np.concatenate([r.sharpes for r in rnd])
        at = np.concatenate([r.alpha_t for r in rnd])
        rows.append({"strategy": "random scores (6 seeds x 3 phases)", "sharpe_mean": round(float(sh.mean()), 2),
                     "sharpe_min": round(float(sh.min()), 2), "sharpe_max": round(float(sh.max()), 2),
                     "cagr_mean": round(float(np.mean([c for r in rnd for c in r.cagrs])), 3),
                     "maxdd_worst": round(float(np.min([d for r in rnd for d in r.dds])), 3),
                     "alpha_yr": round(float(np.mean([a for r in rnd for a in r.alpha])), 3),
                     "alpha_t_mean": round(float(at.mean()), 2), "alpha_t_max": round(float(at.max()), 2)})
        rows += index_rows(ctx.dates, i0)
        df = pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.lit(mode).alias("setup"))
        all_rows.append(df)
        print(f"\n== {mode} ==", flush=True)
        with pl.Config(tbl_rows=20, tbl_width_chars=190, fmt_str_lengths=40):
            print(df.drop("setup"))
    pl.concat(all_rows, how="diagonal").write_csv(DATA / "clean" / "compare_results.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
